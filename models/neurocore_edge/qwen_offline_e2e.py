#!/usr/bin/env python3
"""Offline real-weight Qwen3 Hindi multi-role smoke test (no API, no training)."""
from __future__ import annotations
import json, os, re, sys, time
from datetime import datetime, timezone
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

MODEL_ID = os.environ.get("QWEN_MODEL_ID", "Qwen/Qwen3-0.6B")
OUT = Path(os.environ.get("QWEN_REPORT_DIR", "qwen-report"))
OUT.mkdir(parents=True, exist_ok=True)

SOURCE = """भारत सरकार, शिक्षा मंत्रालय की समग्र शिक्षा की संशोधित पुस्तकालय/पठन दिशा-निर्देश (28 अक्टूबर 2021) के इस परीक्षण-सार में:
1. प्रत्येक कक्षा के लिए प्रति सप्ताह कम-से-कम दो पुस्तकालय पीरियड निर्धारित किए जाएँ।
2. अलग पुस्तकालय कक्ष उपलब्ध न हो तो कक्षा में reading corner / पठन-कोना बनाया जा सकता है।
3. Readers' Clubs और Topic Circles तथा पढ़ने की गतिविधियों को प्रोत्साहित किया जाता है।
4. इस दिए गए स्रोत-सार में राजस्थान/स्थानीय अनुदान की पात्रता, राशि या आवेदन-प्रक्रिया नहीं दी गई है। ऐसी जानकारी गढ़नी नहीं है; उसे अज्ञात बताना है।
स्रोत URL: https://samagra.education.gov.in/docs/revised_guidelines_for_promotion_Library_Reading%20.pdf
"""
TASK = """स्कूल में पुस्तकालय और पढ़ने की संस्कृति बेहतर करने के लिए हिंदी में एक तथ्य-आधारित योजना बनानी है।
ऊपर दिए गए SOURCE को ही तथ्यात्मक आधार मानें। मुख्य शर्तें:
- प्रति कक्षा हर सप्ताह कम-से-कम दो library periods शामिल हों।
- पुस्तकालय कक्ष नहीं है तो reading corner का विकल्प दें।
- Readers' Club / Topic Circles और नियमित पठन गतिविधियाँ शामिल हों।
- राजस्थान/स्थानीय grant rules, पात्रता या राशि के बारे में कोई अनुमान न लगाएँ; स्रोत में न होने पर साफ़ 'स्रोत में जानकारी नहीं / सत्यापन आवश्यक' कहें।
किसी भी तथ्य को स्रोत से अधिक मजबूत या अलग न बताएं।
"""

def ask(tokenizer, model, system: str, user: str, max_new_tokens: int = 180) -> dict:
    messages = [{"role":"system","content":system},{"role":"user","content":user}]
    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_tokens = int(inputs["input_ids"].shape[-1])
    start = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            repetition_penalty=1.05, eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    elapsed = round(time.perf_counter()-start, 3)
    new_tokens = generated[0, input_tokens:]
    output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return {"output":output,"input_tokens":input_tokens,"generated_tokens":int(new_tokens.numel()),
            "generation_seconds":elapsed,
            "tokens_per_second":round(int(new_tokens.numel())/elapsed,2) if elapsed else None}

def verify(text: str) -> dict:
    t = re.sub(r"\s+", " ", text.lower())
    two_periods = bool(
        re.search(r"(दो|2).{0,55}(पुस्तकालय|लाइब्रेरी|library).{0,55}(सप्ताह|week|weekly)", t)
        or re.search(r"(पुस्तकालय|लाइब्रेरी|library).{0,55}(दो|2).{0,35}(सप्ताह|week|weekly)", t))
    reading_corner = bool(re.search(r"(पठन[\s-]?कोना|रीडिंग[\s-]?कॉर्नर|reading corner)", t))
    readers_club = bool(re.search(r"(पाठक[\s-]?क्लब|रीडर्स?[\s-]?क्लब|readers['’]? club|topic circles?)", t))
    local_grant_unknown = bool(
        re.search(r"(स्थानीय|राजस्थान|राज्य|local|rajasthan).{0,90}(अनुदान|grant).{0,90}(नहीं|अज्ञात|उल्लेख|जानकारी|पुष्टि|सत्यापन|unknown|not specified|not provided)", t)
        or re.search(r"(अनुदान|grant).{0,90}(स्रोत में नहीं|उल्लेख नहीं|अज्ञात|सत्यापन आवश्यक|unknown|not specified|not provided)", t))
    return {"mentions_two_library_periods_per_week":two_periods,
            "mentions_reading_corner":reading_corner,
            "mentions_readers_club_or_topic_circles":readers_club,
            "marks_local_grant_rules_as_unverified":local_grant_unknown,
            "all_outputs_nonempty":bool(text.strip())}

def main() -> int:
    torch.set_num_threads(min(2, os.cpu_count() or 1))
    set_seed(20261124)
    started = time.perf_counter()
    print(f"Loading public model weights: {MODEL_ID}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=False)
    model.eval()
    print("Model loaded; running sequential Research -> Planning -> Hindi roles.", flush=True)
    research = ask(tokenizer, model,
        "आप Research Agent हैं। दिए गए स्रोत-सार से केवल समर्थित तथ्य निकालें। कमियाँ और अज्ञात बातें स्पष्ट करें। उत्तर हिंदी में दें।",
        "SOURCE:\n"+SOURCE+"\nTASK:\n"+TASK+"\nकेवल संक्षिप्त शोध निष्कर्ष दें।", 180)
    planning = ask(tokenizer, model,
        "आप Planning Agent हैं। Research Agent के निष्कर्षों के आधार पर क्रमबद्ध, व्यावहारिक योजना बनाएं। स्रोत से बाहर के नियम न गढ़ें। हिंदी में उत्तर दें।",
        "मूल TASK:\n"+TASK+"\nResearch Agent output:\n"+research["output"]+"\nअब योजना बनाएं।", 180)
    final = ask(tokenizer, model,
        "आप Hindi Synthesis Agent हैं। Research और Planning को मिलाकर साफ़, उपयोगी हिंदी में अंतिम उत्तर दें। सभी अनिश्चितताओं को बनाए रखें और कोई तथ्य न गढ़ें।",
        "मूल TASK:\n"+TASK+"\nResearch output:\n"+research["output"]+"\nPlanning output:\n"+planning["output"]+"\nअंतिम उत्तर दें।", 220)
    combined = "RESEARCH:\n"+research["output"]+"\nPLANNING:\n"+planning["output"]+"\nFINAL HINDI:\n"+final["output"]
    checks = verify(final["output"])
    quality_pass = all(checks.values())
    report = {
        "model":MODEL_ID,"model_revision":getattr(model.config,"_commit_hash",None),
        "mode":"github_actions_cpu_public_weights_no_api_no_training",
        "created_at_utc":datetime.now(timezone.utc).isoformat(),
        "pipeline":"sequential Research -> Planning -> Hindi; same checkpoint for all three roles",
        "source":"Samagra Shiksha revised library/reading guidelines, 2021-10-28 (summary supplied to test; official PDF URL in prompt)",
        "agents":{"research":research,"planning":planning,"hindi":final},
        "independent_final_answer_checks":checks,"quality_status":"PASS" if quality_pass else "FAIL",
        "elapsed_seconds":round(time.perf_counter()-started,3),
        "limitations":[
            "The three roles are prompts over one Qwen3-0.6B checkpoint, not three separately trained agents.",
            "String-level checks are a screening aid, not a full semantic or source-authenticity audit.",
            "This is inference only; no training, fine-tuning, API endpoint, or API key is used."]}
    (OUT/"qwen_offline_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    md=["# Qwen3-0.6B Hindi multi-role offline test","",
        f"- **Quality status:** {report['quality_status']}",
        f"- **Mode:** {report['mode']}",
        f"- **Elapsed:** {report['elapsed_seconds']} s",
        "- **Roles:** sequential Research → Planning → Hindi, same Qwen checkpoint",
        "","## Independent checks on final Hindi answer",""]
    md += [f"- {'PASS' if v else 'FAIL'} — {k}" for k,v in checks.items()]
    md += ["","## Research output","",research["output"],"","## Planning output","",planning["output"],
           "","## Final Hindi output","",final["output"],"","## Limits",""]
    md += ["- "+x for x in report["limitations"]]
    (OUT/"qwen_offline_report.md").write_text("\n".join(md),encoding="utf-8")
    print(json.dumps({"quality_status":report["quality_status"],"checks":checks,
                      "elapsed_seconds":report["elapsed_seconds"]},ensure_ascii=False),flush=True)
    return 0 if quality_pass else 2

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"QWEN_OFFLINE_TEST_ERROR: {type(exc).__name__}: {exc}",file=sys.stderr,flush=True)
        raise
