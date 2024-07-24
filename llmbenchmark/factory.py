import torch
from easydict import EasyDict as edict
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import ipdb


def build_torch_dtype(type_str: str) -> torch.dtype:
    TorchDTypeMapper = {
        "torch.float32": torch.float32,
        "float32": torch.float32,
        "float": torch.float32,
        "torch.float16": torch.float16,
        "float16": torch.float16,
        "torch.bfloat16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "torch.float": torch.float32,
        "torch.half": torch.float16,
        "half": torch.float16,
        "torch.float64": torch.float64,
        "float64": torch.float64,
        "torch.double": torch.float64,
        "double": torch.float64,
        "torch.long": torch.long,
        "long": torch.long,
        "int": torch.int,
        "torch.int": torch.int,
        "torch.int32": torch.int32,
        "int32": torch.int32,
        "torch.int64": torch.int64,
        "int64": torch.int64,
    }
    return TorchDTypeMapper[type_str]


def build_llm_model(model_cfgs: dict | edict):
    model_cfgs = edict(model_cfgs)
    print(f"[LLMBenchmark] Building Model...")
    print(f"[LLMBenchmark] Options: {model_cfgs.Options}")
    print(f"[LLMBenchmark] Params: {model_cfgs.Params}")

    model = None
    if model_cfgs.Options.load_weights:
        model_cfgs.Params.torch_dtype = build_torch_dtype(model_cfgs.Params.torch_dtype)
        model = AutoModelForCausalLM.from_pretrained(**model_cfgs.Params)
    else:
        raise NotImplementedError("Not loading weights is not supported yet.")

    if model_cfgs.Params.device_map is None:
        model = model.to(model_cfgs.Options.device)

    print(f"[LLMBenchmark] Model built.")
    return model


def build_tokenizer(tokenizer_cfgs: dict | edict):
    tokenizer_cfgs = edict(tokenizer_cfgs)
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=tokenizer_cfgs.Params.pretrained_model_name_or_path,
        cache_dir=tokenizer_cfgs.Params.cache_dir,
        trust_remote_code=tokenizer_cfgs.Params.trust_remote_code,
    )
    return tokenizer


def build_llm_model_and_tokenizer(
    model_cfgs: dict | edict, tokenizer_cfgs: dict | edict
):
    model = build_llm_model(model_cfgs)
    tokenizer = build_tokenizer(tokenizer_cfgs)
    return model, tokenizer
