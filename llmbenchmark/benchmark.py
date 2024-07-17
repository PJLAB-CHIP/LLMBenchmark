from argparse import ArgumentParser
from ruamel.yaml import YAML
import time
import torch
from transformers import AutoTokenizer
from easydict import EasyDict as edict
from pandas import DataFrame as df
import sys
import os


from llmbenchmark.factory import build_llm_model_and_tokenizer
from llmbenchmark.gen_input_token import gen_random_input_token

RESULT = df(
    columns=[
        "model",
        "hardware",
        "prompt_size",
        "batch_size",
        "token_size",
        "prompt_time",
        "token_time",
        "e2e_time",
        "tensor_parallel",
    ]
)


def measure_inference_times(model, input_ids, token_size, device):
    # Move inputs to device
    input_ids = input_ids.to(device)  # Shape: (batch_size, prompt_size)

    # Measure prompt time
    start_time = time.time()
    with torch.no_grad():
        outputs = model(input_ids)
    prompt_time = time.time() - start_time

    # Prepare for next token generation
    past_key_values = outputs.past_key_values
    next_token = input_ids[:, -1].unsqueeze(1)

    # Measure token time for the next `token_size` tokens
    token_times = []
    for _ in range(token_size - 1):
        start_time = time.time()
        with torch.no_grad():
            outputs = model(next_token, past_key_values=past_key_values)
        token_time = time.time() - start_time
        token_times.append(token_time)

        # Update next token and past_key_values for the next iteration
        past_key_values = outputs.past_key_values
        next_token = outputs.logits.argmax(dim=-1)

    avg_token_time = sum(token_times) / len(token_times)

    return prompt_time * 1e3, avg_token_time * 1e3  # ms


def main(cfgs: dict | edict):
    cfgs = edict(cfgs)

    # Set $CUDA_VISIBLE_DEVICES.
    os.environ["CUDA_VISIBLE_DEVICES"] = cfgs.Benchmark.CUDA_VISIBLE_DEVICES
    
    # Create result dir.
    if not os.path.exists(os.path.dirname(cfgs.Benchmark.result_path)):
        os.makedirs(os.path.dirname(cfgs.Benchmark.result_path))

    device = torch.device(cfgs.Benchmark.device)

    # Build model.
    model, tokenizer = build_llm_model_and_tokenizer(cfgs.Model, cfgs.Tokenizer)

    batch_size = list(cfgs.Benchmark.batch_size)
    prompt_size = list(cfgs.Benchmark.prompt_size)
    token_size = list(cfgs.Benchmark.token_size)
    vocab_size = tokenizer.vocab_size

    # Heat up the model.
    input_ids = gen_random_input_token(1, 128, vocab_size)
    _, _ = measure_inference_times(model, input_ids, 128, device)

    i = 1
    n_iters = len(batch_size) * len(prompt_size) * len(token_size)

    try:
        for bs in batch_size:
            for ps in prompt_size:
                for ts in token_size:
                    # Generate random input tokens.
                    input_ids = gen_random_input_token(bs, ps, vocab_size)
                    # Measure inference times.
                    prompt_time, avg_token_time = measure_inference_times(
                        model, input_ids, ts, device
                    )
                    e2e_time = prompt_time + avg_token_time * (ts - 1)
                    # Save results.
                    RESULT.loc[len(RESULT)] = [
                        cfgs.Model.pretrained_model_name_or_path,  # Model
                        cfgs.Benchmark.hardware_name,  # Hardware Name
                        ps,  # Prompt size
                        bs,  # Batch size
                        ts,  # Token size
                        prompt_time,  # Prompt time
                        avg_token_time,  # Token time
                        e2e_time,  # E2E time
                        torch.cuda.device_count(),  # Tensor parallel
                    ]
                    # Print result of current iteration.
                    print(
                        f"{i}/{n_iters}: "
                        f"{RESULT.tail(1).to_string(index=False, header=False)}"
                    )
                    sys.stdout.flush()
                    i = i + 1
    except Exception as e:
        print(f"[LLMBenchmark] Fatal Error: {e}")

    # Create a new file with header if there not existing.
    if not os.path.exists(cfgs.Benchmark.result_path):
        RESULT.to_csv(cfgs.Benchmark.result_path, index=False)
    # Otherwise append to the existing file.
    else:
        RESULT.to_csv(cfgs.Benchmark.result_path, mode="a", header=False, index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to the config file."
    )
    args = parser.parse_args()
    cfgs = YAML().load(open(args.config))
    main(cfgs)