# LLMBenchmark

## Environment Setup

Install pytorch based on [official website](https://pytorch.org).

Install other dependencies with following command:

```bash
pip install -v -e .
```

## Quick Start

Create a YAML configure file (or use one of we provide under [./configs](./configs/) directory).

Start benchmark with following command:

```bash
python llmbenchmark/benchmark.py -c <path_to_config_file>
```

## Model Zoo

<p align="center">

| Model Name | Config File |
|:---:|:---:|
| llama-7b | [llama-7b-hf.yml](./configs/llama-7b-hf.yml) |
| opt-6.7b | [opt-6.7b.yml](./configs/opt-6.7b.yml) |
| opt-13b | [opt-13b.yml](./configs/opt-6.7b.yml) |
| opt-66b | [opt-66b.yml](./configs/opt-6.7b.yml) |

</p>
