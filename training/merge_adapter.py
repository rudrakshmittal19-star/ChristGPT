"""Merge the LoRA adapter into the base model (for GGUF/Ollama/vLLM deployment). Run: python -m training.merge_adapter"""
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from campusgpt.config import ADAPTER, BASE_MODEL, ROOT

out = ROOT / "training" / "outputs" / "campusgpt-merged"
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16)
model = PeftModel.from_pretrained(model, ADAPTER).merge_and_unload()
model.save_pretrained(out)
AutoTokenizer.from_pretrained(BASE_MODEL).save_pretrained(out)
print("Merged model saved to", out)
