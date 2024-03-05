# A Multi-Level Framework for Accelerating Training Transformer Models

## Brief Introduction

This is the repository for multi-level training framework. We provide the Coalescing, De-coalescing and Interpolation operators described in the paper and an example of accelerating the pre-training of GPT-2 on Wiki-En. The framework is built based on the [transformers](https://github.com/huggingface/transformers).

## Install Requirements

**Step 1**

To use the map_tools for the model mapping, please install following packages.

```bash
pip install torch==2.0.1+cu118 transformers==4.31.0
```

**Step 2**

If you hope to run the pre-training acceleration example, please install following packages.

```bash
pip install datasets==2.14.6 evaluate==0.4.0
```

## Map Tools Usage

Please refer to map_tools/README.md .

## Accelerate Pre-training of GPT-2 on Wikipedia-En

wait for implementation...

## Citation

Please cite our paper if you find the repo helpful for you:

```bibtex
@inproceedings{
    zou2024a,
    title={A Multi-Level Framework for Accelerating Training Transformer Models},
    author={Longwei Zou and Han Zhang and Yangdong Deng},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024},
    url={https://openreview.net/forum?id=BI1N3lTWtn}
}
```