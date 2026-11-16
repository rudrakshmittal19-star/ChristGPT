"""QLoRA fine-tuning of a small instruct model on data/qa/train.jsonl (works on a free Colab T4).

Run:  python -m training.train_qlora --epochs 2
Output: LoRA adapter in training/outputs/campusgpt-lora + loss_curve.png
"""
import argparse
import inspect
import json

import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from campusgpt.config import ADAPTER, BASE_MODEL, QA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--data", default=str(QA / "train.jsonl"))
    ap.add_argument("--out", default=ADAPTER)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--r", type=int, default=16)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    tok.pad_token = tok.pad_token or tok.eos_token
    ds = load_dataset("json", data_files=a.data, split="train")
    ds = ds.map(lambda ex: {"text": tok.apply_chat_template(ex["messages"], tokenize=False)}, remove_columns=ds.column_names)
    print(f"{len(ds)} training examples")

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.float16)  # T4 has no bf16
    model = AutoModelForCausalLM.from_pretrained(a.model, quantization_config=bnb, device_map="auto")
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(r=a.r, lora_alpha=2 * a.r, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

    key = "max_length" if "max_length" in inspect.signature(SFTConfig).parameters else "max_seq_length"
    cfg = SFTConfig(output_dir=a.out, num_train_epochs=a.epochs, per_device_train_batch_size=a.batch,
                    gradient_accumulation_steps=a.grad_accum, learning_rate=a.lr, lr_scheduler_type="cosine",
                    warmup_ratio=0.05, logging_steps=5, save_strategy="no", fp16=True, optim="paged_adamw_8bit",
                    gradient_checkpointing=True, report_to="none", dataset_text_field="text", **{key: a.max_len})
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, peft_config=lora,
                         **({"processing_class": tok} if "processing_class" in inspect.signature(SFTTrainer).parameters else {"tokenizer": tok}))
    trainer.train()
    trainer.model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    logs = [l for l in trainer.state.log_history if "loss" in l]
    json.dump(logs, open(f"{a.out}/train_log.json", "w"))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.plot([l["step"] for l in logs], [l["loss"] for l in logs])
        plt.xlabel("step"); plt.ylabel("training loss"); plt.title("QLoRA training loss"); plt.grid(alpha=.3)
        plt.savefig(f"{a.out}/loss_curve.png", dpi=150)
    except Exception as e:
        print("Could not plot loss:", e)
    print("Adapter saved to", a.out)


if __name__ == "__main__":
    main()
