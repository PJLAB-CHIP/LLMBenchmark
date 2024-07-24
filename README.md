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

Anything supported by hugging face.