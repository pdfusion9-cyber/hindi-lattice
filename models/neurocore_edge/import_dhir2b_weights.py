import argparse,json,torch
from dhir2b_neurocore import Dhir2BNeuroCore
def main():
 p=argparse.ArgumentParser(description="Import pinned Dhīr-2B weights into NeuroCore on CPU")
 p.add_argument("--revision",required=True); p.add_argument("--output",required=True)
 p.add_argument("--model-id",default="zenteiq-ai/Dhir-2B"); p.add_argument("--rank",type=int,default=8)
 p.add_argument("--dtype",choices=("bfloat16","float32"),default="bfloat16")
 p.add_argument("--threads",type=int,default=4); a=p.parse_args()
 torch.set_num_threads(a.threads)
 dtype=torch.bfloat16 if a.dtype=="bfloat16" else torch.float32
 model,tokenizer=Dhir2BNeuroCore.import_pretrained(a.model_id,a.revision,a.rank,dtype)
 report=model.save_checkpoint(a.output,tokenizer); report["device"]="cpu"; report["gpu_used"]=False
 print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
