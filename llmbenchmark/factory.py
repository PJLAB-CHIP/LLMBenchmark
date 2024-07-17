import torch
from easydict import EasyDict as edict
from transformers import AutoModelForCausalLM, AutoTokenizer


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


def build_llm_model_and_tokenizer(
    model_cfgs: dict | edict, tokenizer_cfgs: dict | edict
):
    model_cfgs = edict(model_cfgs)
    tokenizer_cfgs = edict(tokenizer_cfgs)

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_cfgs.pretrained_model_name_or_path,
        cache_dir=model_cfgs.cache_dir,
        torch_dtype=build_torch_dtype(model_cfgs.torch_dtype),
        trust_remote_code=model_cfgs.trust_remote_code,
        device_map=model_cfgs.device_map,
    )

    if model_cfgs.device_map is None:
        model = model.to(model_cfgs.device)

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=tokenizer_cfgs.pretrained_model_name_or_path,
        cache_dir=tokenizer_cfgs.cache_dir,
        trust_remote_code=tokenizer_cfgs.trust_remote_code,
    )

    return model, tokenizer
