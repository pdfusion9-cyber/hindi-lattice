"""Manual live-provider Hindi E2E smoke test. Synthetic task and public source only."""
import argparse, json, os, sys, time, urllib.request, urllib.error

SOURCE={"ref":"samagra-library-guidelines-2021","date":"2021-10-28","url":"https://samagra.education.gov.in/docs/revised_guidelines_for_promotion_Library_Reading%20.pdf","facts":["At least two library periods per week for every class.","A reading corner may be created if there is no separate library room.","Readers' Clubs or Topic Circles and reading/writing activities are encouraged."]}
TASK="राजस्थान के छोटे सरकारी प्राथमिक विद्यालय के लिए पुस्तकालय/पठन-कोना और पाठक क्लब की हिंदी कार्ययोजना बनाएं। स्रोत से बाहर के स्थानीय अनुदान/खरीद नियम अपुष्ट बताएं।"

def call(base,key,model,system,user):
    url=base.rstrip("/")
    if not url.endswith("/chat/completions"): url+="/chat/completions"
    body=json.dumps({"model":model,"messages":[{"role":"system","content":system},{"role":"user","content":user}],"temperature":0.1,"max_tokens":700,"response_format":{"type":"json_object"}},ensure_ascii=False).encode()
    req=urllib.request.Request(url,data=body,headers={"Authorization":"Bearer "+key,"Content-Type":"application/json"})
    try:
        with urllib.request.urlopen(req,timeout=45) as r: data=json.loads(r.read(1000000))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"provider HTTP {e.code}: {e.read(600).decode(errors='replace')}") from None
    except (urllib.error.URLError,TimeoutError) as e: raise RuntimeError(f"provider connection: {type(e).__name__}") from None
    msg=data["choices"][0]["message"]["content"]
    try: parsed=json.loads(msg)
    except Exception: raise RuntimeError("provider returned invalid JSON") from None
    if not isinstance(parsed,dict): raise RuntimeError("provider JSON root is not an object")
    return parsed,{"prompt_tokens":data.get("usage",{}).get("prompt_tokens"),"completion_tokens":data.get("usage",{}).get("completion_tokens")}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--base-url",default=os.getenv("LLM_BASE_URL",""))
    p.add_argument("--model",default=os.getenv("LLM_MODEL",""))
    p.add_argument("--report-path",default="live_hindi_e2e_report.json")
    a=p.parse_args(); key=os.getenv("LLM_API_KEY",""); start=time.monotonic(); usage={}
    if not (key and a.base_url and a.model):
        print("LIVE_TEST_NOT_CONFIGURED: set LLM_API_KEY, LLM_BASE_URL and LLM_MODEL"); return 2
    source=json.dumps(SOURCE,ensure_ascii=False)
    try:
        research,usage["research"]=call(a.base_url,key,a.model,
          "आप Research Agent हैं। केवल दिए गए स्रोत से तथ्य निकालें। JSON: findings array (claim,evidence_ref), gaps array.",
          f"कार्य: {TASK}\nस्रोत: {source}")
        findings=research.get("findings")
        if not isinstance(findings,list) or not findings or any(x.get("evidence_ref")!=SOURCE["ref"] for x in findings):
            raise RuntimeError("Research Agent returned missing/unknown evidence IDs")
        plan,usage["planning"]=call(a.base_url,key,a.model,
          "आप Planning Agent हैं। Research findings पर निर्भर क्रमबद्ध सुरक्षित योजना बनाएं। JSON: goal,steps,risks,success_criteria,unverified_local_rules.",
          f"कार्य: {TASK}\nResearch: {json.dumps(research,ensure_ascii=False)}\nस्रोत: {source}")
        if not isinstance(plan.get("steps"),list) or not plan["steps"]: raise RuntimeError("Planning Agent returned no steps")
        hindi,usage["hindi"]=call(a.base_url,key,a.model,
          "आप Hindi Agent हैं। Research और Planning को मिलाकर स्पष्ट स्वाभाविक हिंदी उत्तर दें। स्रोत से बाहर के तथ्य न जोड़ें। JSON: answer,evidence_refs.",
          f"कार्य: {TASK}\nResearch: {json.dumps(research,ensure_ascii=False)}\nPlanning: {json.dumps(plan,ensure_ascii=False)}\nस्रोत: {source}")
        if not isinstance(hindi.get("answer"),str) or len(hindi["answer"])<80 or SOURCE["ref"] not in hindi.get("evidence_refs",[]):
            raise RuntimeError("Hindi Agent answer/source reference invalid")
        verify,usage["independent_verifier"]=call(a.base_url,key,a.model,
          "आप स्वतंत्र Evidence Verifier हैं। केवल source और outputs की तुलना करें। संदेह हो तो passed=false। JSON: passed boolean, unsupported_claims array, missing_requirements array, notes array.",
          f"SOURCE: {source}\nRESEARCH: {json.dumps(research,ensure_ascii=False)}\nPLAN: {json.dumps(plan,ensure_ascii=False)}\nHINDI: {json.dumps(hindi,ensure_ascii=False)}\nजाँचें: हर कक्षा में कम-से-कम दो library periods/week, reading corner, Readers' Club/Topic Circles, और स्थानीय नियमों को अपुष्ट बताना।")
        if verify.get("passed") is not True or verify.get("unsupported_claims") or verify.get("missing_requirements"):
            raise RuntimeError("Independent verifier rejected output: "+json.dumps(verify,ensure_ascii=False)[:600])
        report={"status":"PASS","mode":"live_provider_synthetic_public_source_only","model":a.model,"source":SOURCE,"task":TASK,"agent_order":["research","planning","hindi","independent_verifier"],"research":research,"planning":plan,"hindi":hindi,"verifier":verify,"usage":usage,"elapsed_seconds":round(time.monotonic()-start,2),"limitations":["One live smoke task, not a statistical benchmark.","Verifier uses the configured provider; human audit remains separate.","No model training, dataset ingestion or external action performed."]}
    except Exception as e:
        report={"status":"FAIL","mode":"live_provider_synthetic_public_source_only","model":a.model,"error":f"{type(e).__name__}: {str(e)[:600]}","usage":usage,"elapsed_seconds":round(time.monotonic()-start,2)}
        with open(a.report_path,"w",encoding="utf-8") as f: json.dump(report,f,ensure_ascii=False,indent=2)
        print(json.dumps(report,ensure_ascii=False,indent=2)); return 1
    with open(a.report_path,"w",encoding="utf-8") as f: json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps(report,ensure_ascii=False,indent=2)); return 0

if __name__=="__main__": sys.exit(main())
