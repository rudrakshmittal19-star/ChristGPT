"""One generate() function over several backends: anthropic, openai-compatible (Ollama/vLLM), local HF, HF + LoRA."""

from . import config as C

_hf = {}


def _split(messages):
    system = "\n".join(m["content"] for m in messages if m["role"] == "system")
    return system, [m for m in messages if m["role"] != "system"]


def _hf_model(lora: bool):
    key = "lora" if lora else "base"
    if key not in _hf:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(C.BASE_MODEL)
        kw = (
            dict(torch_dtype=torch.float16, device_map="auto")
            if torch.cuda.is_available()
            else {}
        )
        model = AutoModelForCausalLM.from_pretrained(C.BASE_MODEL, **kw)
        if lora:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, C.ADAPTER)
        model.eval()
        _hf[key] = (tok, model)
    return _hf[key]


def generate(messages, backend=None, max_tokens=400, temperature=0.1) -> str:
    backend = backend or C.LLM_BACKEND
    if backend == "anthropic":
        import anthropic

        system, rest = _split(messages)
        r = anthropic.Anthropic().messages.create(
            model=C.ANTHROPIC_MODEL,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=rest,
        )
        return "".join(b.text for b in r.content if b.type == "text")
    if backend == "groq":
        import time
        from openai import OpenAI, RateLimitError

        client = OpenAI(base_url=C.GROQ_BASE_URL, api_key=C.GROQ_API_KEY)
        for wait in (1, 3, 8, 20):
            try:
                r = client.chat.completions.create(
                    model=C.GROQ_MODEL,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return r.choices[0].message.content
            except RateLimitError:
                time.sleep(wait)
        raise RuntimeError("Groq rate limit: try again in a minute")
    if backend == "openai":
        from openai import OpenAI

        r = OpenAI(base_url=C.OPENAI_BASE_URL, api_key="local").chat.completions.create(
            model=C.OPENAI_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return r.choices[0].message.content
    if backend in ("hf", "hf-lora"):
        import torch

        tok, model = _hf_model(backend == "hf-lora")
        prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                pad_token_id=tok.eos_token_id,
            )
        return tok.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()
    raise ValueError(
        f"Backend '{backend}' cannot generate text (use anthropic, openai, hf or hf-lora)"
    )


def generate_stream(messages, backend=None, max_tokens=400, temperature=0.1):
    backend = backend or C.LLM_BACKEND
    if backend in ("groq", "openai"):
        from openai import OpenAI, RateLimitError
        import time

        client = (
            OpenAI(base_url=C.GROQ_BASE_URL, api_key=C.GROQ_API_KEY)
            if backend == "groq"
            else OpenAI(base_url=C.OPENAI_BASE_URL, api_key="local")
        )
        model = C.GROQ_MODEL if backend == "groq" else C.OPENAI_MODEL
        stream = None
        for wait in (1, 3, 8, 20) if backend == "groq" else (0,):
            try:
                stream = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True,
                )
                break
            except RateLimitError:
                if wait == 20:
                    raise RuntimeError("Groq rate limit: try again in a minute")
                time.sleep(wait)
        for chunk in stream:
            text = chunk.choices[0].delta.content
            if text:
                yield text
        return
    text = generate(
        messages, backend=backend, max_tokens=max_tokens, temperature=temperature
    )
    for offset in range(0, len(text), 32):
        yield text[offset : offset + 32]
