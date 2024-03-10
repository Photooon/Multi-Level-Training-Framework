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

If you hope to run the pre-training acceleration example, please install packages as follows.

```bash
cd example
pip install -r requirements.txt
```

## Map Tools Usage

We implemented the three operators to ochestrate the multi-level training framework in map_tools. With map tools, it's convenient to resize and merge transformers. The usage of map tools could be found in [map_tools document](map_tools/README.md).

## Accelerate Pre-training of GPT-2 on Wikipedia-En

To better illustrate the usage of map tools and demonstrate the effectiveness of the multi-level training framework, we provide a example of accelerating the GPT-2 pre-training on Wikipedia-En. Our framework could save around 23.5% training time for it in our testing environment of 8 * NVIDIA A100 (Note that the speedup ratio depends on datasets, models and machines.).

If you hope to run the example, 150GB space of disk is required to preprocess the wikipedia dataset. Additionally, around 1.5 days is needed to run through the example with 8 * NVIDIA A100. 

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