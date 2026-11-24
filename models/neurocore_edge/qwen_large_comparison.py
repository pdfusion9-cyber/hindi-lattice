#!/usr/bin/env python3
"""Comparable Qwen3-1.7B / 4B GGUF CPU benchmark; no API and no training."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

LLAMA_CLI = os.environ.get("LLAMA_CLI", "llama.cpp/build/bin/llama-cli")
OUT = Path(os.environ.get("QWEN_REPORT_DIR", "qwen-large-report"))
OUT.mkdir(parents=True, exist_ok=True)
THREADS = os.environ.get("LLAMA_THREADS", "4")
CTX = os.environ.get("LLAMA_CTX_SIZE", "8192")

SOURCE_CONTEXT = (
    "स्रोत तथ्य: हर कक्षा में सप्ताह में कम-से-कम दो पुस्तकालय अवधियाँ। "
    "अलग कमरा न हो तो पठन-कोना। पाठक क्लब और कहानी-पठन गतिविधियाँ। "
    "स्थानीय अनुदान नियम इस स्रोत में नहीं हैं।"
)
TASK = (
    "राजस्थान के छोटे सरकारी प्राथमिक विद्यालय में पुस्तकालय और पाठक क्लब शुरू करने "
    "की कार्ययोजना बनाओ। केवल दिए तथ्य लो।"
)
SOURCE_URL = "https://samagra.education.gov.in/docs/revised_guidelines_for_promotion_Library_Reading%20.pdf"

MODELS = [
    {"name": "Qwen3-1.7B", "repo": "Qwen/Qwen3-1.7B-GGUF", "quant": "Q4_K_M"},
    {"name": "Qwen3-4B", "repo": "Qwen/Qwen3-4B-GGUF", "quant": "Q4_K_M"},
]

V5_BASELINE = {
    "model": "Anviksha V5 Hindi checkpoint",
    "source_context": SOURCE_CONTEXT,
    "task": TASK,
    "mode": "local CPU checkpoint inference; no API/no training",
    "pipeline": "sequential Research -> Planning -> Hindi; single V5 checkpoint for all roles",
    "outputs": {
        "research": "<think>सामा",
        "planning": "<think>सामान्य। उत्तर: योजना बनाओ। पशोध निष्कर्ष: उन-अव",
        "hindi": "<think><ans>योजना: पीठ।\\n\\nगत-शोध: ग्रे स्थानीय",
    },
    "checks": {
        "mentions_two_library_periods": False,
        "mentions_reading_corner": False,
        "mentions_reader_club": False,
        "states_local_grant_rules_unknown": False,
        "all_outputs_nonempty": True,
    },
    "elapsed_seconds": 3.04,
    "comparison_note": (
        "This is the prior actual V5 run on the same source-context and task. "
        "It used a shorter legacy role prompt and a different inference runtime; "
        "use it as a historical baseline, not a controlled same-binary comparison."
    ),
}

def run_role(model_ref: str, system: str, user: str, max_tokens: int) -> dict:
    args = [
        LLAMA_CLI,
        "-hf", model_ref,
        "-ngl", "0",
        "-t", THREADS,
        "-c", CTX,
        "-n", str(max_tokens),
        "--temp", "0.7",
        "--top-k", "20",
        "--top-p", "0.8",
        "--min-p", "0",
        "--presence-penalty", "1.5",
        "--repeat-penalty", "1.05",
        "--jinja",
        "--reasoning", "off",
        "--system-prompt", system,
        "--prompt", user,
        "--single-turn",
        "--simple-io",
        "--no-display-prompt",
        "--no-show-timings",
        "--no-warmup",
        "--log-verbosity", "1",
    ]
    started = time.perf_counter()
    completed = subprocess.run(
        args, capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=1800, check=False,
    )
    elapsed = round(time.perf_counter() - started, 3)
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    if completed.returncode != 0:
        raise RuntimeError(
            f"llama-cli exited {completed.returncode}; stderr tail: {stderr[-6000:]}; "
            f"stdout tail: {stdout[-2000:]}"
        )
    # Keep only clean generated text; simple-io/no-display-prompt should omit the input.
    output = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", stdout).strip()
    if not output:
        raise RuntimeError(f"llama-cli returned an empty answer. stderr tail: {stderr[-4000:]}")
    return {
        "output": output,
        "wall_seconds": elapsed,
        "output_characters": len(output),
        "output_words": len(output.split()),
        "max_new_tokens": max_tokens,
        "return_code": completed.returncode,
        "stderr_tail": stderr[-1200:],
    }

def _lines(text: str):
    return [re.sub(r"\s+", " ", x.strip().lower()) for x in text.splitlines() if x.strip()]

def verify(text: str) -> dict:
    lines = _lines(text)
    all_text = " ".join(lines)

    def line_has(items, line):
        return any(item in line for item in items)

    library_terms = ["पुस्तकालय", "लाइब्रेरी", "library"]
    period_terms = ["अवधि", "पीरियड", "सत्र", "कालांश", "period", "periods"]
    two_terms = ["दो", "2", "२", "two"]
    week_terms = ["सप्ताह", "साप्ताहिक", "हफ्ते", "हफ्ते में", "प्रति सप्ताह", "weekly", "per week", "week"]
    mentions_two_periods = any(
        line_has(library_terms, line)
        and line_has(period_terms, line)
        and line_has(two_terms, line)
        and line_has(week_terms, line)
        for line in lines
    )
    reading_corner = any(
        term in all_text for term in [
            "पठन-कोना", "पठन कोना", "रीडिंग कॉर्नर", "reading corner", "कक्षा में कोना",
        ]
    )
    readers_club = any(
        term in all_text for term in [
            "पाठक क्लब", "पाठक-क्लब", "रीडर्स क्लब", "readers' club",
            "readers club", "topic circles", "विषय मंडली", "पठन क्लब",
        ]
    )
    local_terms = ["स्थानीय", "राजस्थान", "राज्य", "अनुदान", "grant", "पात्रता", "राशि"]
    unknown_terms = [
        "स्रोत में नहीं", "उल्लेख नहीं", "जानकारी नहीं", "उपलब्ध नहीं",
        "अज्ञात", "सत्यापन", "पुष्टि करें", "जांच आवश्यक", "जाँच आवश्यक",
        "पता नहीं", "नहीं दी गई", "unknown", "not specified", "not provided",
        "verify locally", "verification",
    ]
    grant_unknown = (
        any(x in all_text for x in local_terms)
        and any(x in all_text for x in unknown_terms)
    )
    return {
        "mentions_two_library_periods_per_week": mentions_two_periods,
        "mentions_reading_corner": reading_corner,
        "mentions_readers_club": readers_club,
        "states_local_grant_rules_unverified": grant_unknown,
        "answer_nonempty": bool(text.strip()),
    }

def run_model(model: dict) -> dict:
    model_ref = model["repo"] + ":" + model["quant"]
    role_results = {}
    research = run_role(
        model_ref,
        "आप Research Agent हैं। नीचे दिए स्रोत तथ्य को स्पष्ट बिंदुओं में निकालें। "
        "समय/आवृत्ति सहित सभी आवश्यक तथ्य रखें। स्रोत में जो नहीं है उसे अज्ञात बताएं। "
        "किसी योजना, राशि या सरकारी पात्रता का अनुमान न लगाएं। हिंदी में उत्तर दें।",
        "स्रोत तथ्य:\n" + SOURCE_CONTEXT + "\nकार्य:\n" + TASK +
        "\nशोध निष्कर्ष क्रमबद्ध और पूर्ण रूप में दें।",
        600,
    )
    role_results["research"] = research

    planning = run_role(
        model_ref,
        "आप Planning Agent हैं। Research Agent के निष्कर्षों का उपयोग कर छोटे सरकारी "
        "प्राथमिक विद्यालय के लिए एक लागू करने योग्य, चरणबद्ध कार्ययोजना बनाएं। दिए गए "
        "स्रोत से बाहर तथ्य न जोड़ें। स्थानीय अनुदान/नियमों को सत्यापन योग्य अज्ञात मानें। "
        "हिंदी में स्पष्ट और पूर्ण उत्तर दें।",
        "मूल कार्य:\n" + TASK + "\nस्रोत तथ्य:\n" + SOURCE_CONTEXT +
        "\nResearch Agent का निष्कर्ष:\n" + research["output"] +
        "\nअब क्रमबद्ध योजना बनाएं।",
        800,
    )
    role_results["planning"] = planning

    hindi = run_role(
        model_ref,
        "आप Hindi Synthesis Agent हैं। Research और Planning को मिलाकर उपयोगकर्ता के लिए "
        "साफ, संपूर्ण और व्यावहारिक हिंदी उत्तर लिखें। कोई तथ्य न गढ़ें। कम-से-कम दो "
        "पुस्तकालय अवधियाँ प्रति कक्षा प्रति सप्ताह, पठन-कोना, पाठक क्लब/कहानी-पठन "
        "गतिविधियाँ और स्थानीय अनुदान नियमों की अनिश्चितता स्पष्ट रूप से रखें।",
        "मूल कार्य:\n" + TASK + "\nस्रोत तथ्य:\n" + SOURCE_CONTEXT +
        "\nResearch output:\n" + research["output"] +
        "\nPlanning output:\n" + planning["output"] +
        "\nअब अंतिम उत्तर लिखें।",
        1200,
    )
    role_results["hindi"] = hindi

    checks = verify(hindi["output"])
    role_chain_ok = bool(research["output"] and planning["output"] and hindi["output"])
    quality_pass = all(checks.values()) and role_chain_ok
    return {
        "model": model["name"],
        "model_ref": model_ref,
        "quantization": model["quant"],
        "mode": "llama.cpp CPU inference with public GGUF weights; no API/no training",
        "pipeline": "sequential Research -> Planning -> Hindi using one checkpoint",
        "roles": role_results,
        "pipeline_dependencies": {
            "planning_received_research_output": True,
            "hindi_received_research_and_planning_outputs": True,
        },
        "independent_final_answer_checks": checks,
        "quality_status": "PASS" if quality_pass else "FAIL",
        "total_wall_seconds": round(sum(x["wall_seconds"] for x in role_results.values()), 3),
    }

def main() -> int:
    started = time.perf_counter()
    results = []
    for model in MODELS:
        print(f"Starting {model['name']} ({model['quant']}) on CPU", flush=True)
        result = run_model(model)
        results.append(result)
        print(json.dumps({
            "model": result["model"],
            "quality_status": result["quality_status"],
            "checks": result["independent_final_answer_checks"],
            "wall_seconds": result["total_wall_seconds"],
        }, ensure_ascii=False), flush=True)

    passed = [r["model"] for r in results if r["quality_status"] == "PASS"]
    overall = "PASS" if passed else "FAIL"
    report = {
        "benchmark": "Anviksha Hindi library & Readers Club plan; historical V5 baseline plus controlled Qwen3 size comparison",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_context": SOURCE_CONTEXT,
        "task": TASK,
        "official_source": SOURCE_URL,
        "llama_cpp_cli": LLAMA_CLI,
        "runtime_mode": "GitHub Actions CPU only; public GGUF model weights; no external inference API; no training or fine-tuning",
        "generation_settings": {
            "threads": THREADS, "context_tokens": int(CTX),
            "temperature": 0.7, "top_p": 0.8, "top_k": 20,
            "presence_penalty": 1.5, "reasoning": "off",
            "max_new_tokens": {"research": 600, "planning": 800, "hindi": 1200},
        },
        "historical_v5_baseline": V5_BASELINE,
        "qwen_results": results,
        "overall_qwen_quality_status": overall,
        "models_passing": passed,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "limitations": [
            "Qwen3-1.7B and Qwen3-4B use official Q4_K_M GGUF quantizations; quantization can change output quality.",
            "Both Qwen models share the same prompts, role order, CPU inference engine, and output limits for a controlled size comparison.",
            "The saved V5 baseline was a prior actual inference run on the same source context and task, but used its original role prompts/runtime; it is historical rather than a same-binary controlled rerun.",
            "The three agent roles are prompts over one shared checkpoint per model, not separately trained agent models.",
            "The independent verifier uses phrase/line matching. It is a screening check, not a complete semantic or source-authenticity audit.",
        ],
    }
    (OUT / "qwen_large_comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    md = [
        "# Anviksha Hindi benchmark — V5 baseline vs Qwen3-1.7B / 4B",
        "",
        f"- **Overall Qwen status:** {overall}",
        f"- **Elapsed:** {report['elapsed_seconds']} seconds",
        "- **Inference:** GitHub Actions CPU, llama.cpp, official Q4_K_M GGUF weights",
        "- **No API / no training / no fine-tuning**",
        "",
        "## Model comparison",
        "",
        "| Model | Quantization | Quality | 2 library periods/week | Reading corner | Readers Club | Local grant unknown |",
        "|---|---|---|---|---|---|---|",
    ]
    v5 = V5_BASELINE["checks"]
    md.append(
        "| Anviksha V5 (historical) | existing checkpoint | FAIL | "
        + ("PASS" if v5["mentions_two_library_periods"] else "FAIL") + " | "
        + ("PASS" if v5["mentions_reading_corner"] else "FAIL") + " | "
        + ("PASS" if v5["mentions_reader_club"] else "FAIL") + " | "
        + ("PASS" if v5["states_local_grant_rules_unknown"] else "FAIL") + " |"
    )
    for r in results:
        c = r["independent_final_answer_checks"]
        md.append(
            f"| {r['model']} | {r['quantization']} | {r['quality_status']} | "
            f"{'PASS' if c['mentions_two_library_periods_per_week'] else 'FAIL'} | "
            f"{'PASS' if c['mentions_reading_corner'] else 'FAIL'} | "
            f"{'PASS' if c['mentions_readers_club'] else 'FAIL'} | "
            f"{'PASS' if c['states_local_grant_rules_unverified'] else 'FAIL'} |"
        )
    md += ["", "## Historical V5 outputs", ""]
    for role, output in V5_BASELINE["outputs"].items():
        md += [f"### {role}", "", output, ""]
    for r in results:
        md += [f"## {r['model']} — {r['quality_status']}", ""]
        md += ["### Independent checks", ""]
        md += [f"- {'PASS' if v else 'FAIL'} — {k}" for k, v in r["independent_final_answer_checks"].items()]
        for role, item in r["roles"].items():
            md += ["", f"### {role} output", "", item["output"], "",
                   f"- Wall time: {item['wall_seconds']} s; output: {item['output_words']} words / {item['output_characters']} characters."]
    md += ["", "## Limits", ""] + ["- " + x for x in report["limitations"]]
    (OUT / "qwen_large_comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({
        "overall_qwen_quality_status": overall,
        "models_passing": passed,
        "elapsed_seconds": report["elapsed_seconds"],
    }, ensure_ascii=False), flush=True)
    return 0 if passed else 2

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"QWEN_LARGE_BENCHMARK_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise

# Benchmark revision: trigger the dedicated GitHub Actions workflow after adding its workflow definition.
