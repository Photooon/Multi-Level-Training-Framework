# Accelerating the Pre-training of GPT-2 on Wiki-En

## Preparation

**Accessible to huggingface**

Make sure your can visit the [huggingface](https://huggingface.co/). If not, we recommand you to use a mirror site.

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

**Install packages**

Install required packages to run scripts.

```bash
pip install -r requirements.txt
```

## Data Preprocess

**Preprocess Wikipedia-En**

Data.py would download, tokenize and chunk the text of Wikipedia-En. The defualt max data length is 512. Note that you should prepare around 150GB space on your disk for the dataset and its cache.

```bash
python data.py
```

After that, you could visit the dataset in datasets/gpt2_wikipedia .

## Pre-training

**Pre-training a GPT-2 from scratch**

We first pre-training a GPT-2 from scratch as the baseline. The script has been verified with 8*A100 (80GB). The default setting need around 70GB memory on each GPU. You could modified values of GPU_NUM, PER_DEVICE_BATCH_SIZE and GRADIENT_ACCUMULATION at the beginning of the script according to your environment. The product of them is the global batch size. Ensure the product equals to 512.

```bash
./scripts/pretrain.sh
```

The script will give the running time and the evaluation result on the validation set. About **1144min** are required to pre-train a GPT-2 on 8*A100 from scratch. The validation loss is around 2.718.


**Multi-level pre-training a GPT-2**

Everything for a two levels pre-training of GPT-2 has been assembled in the multilevel_pretrain.sh, including the pre-training and three mapping operation. Remember to modified the values of GPU_NUM, PER_DEVICE_BATCH_SIZE and GRADIENT_ACCUMULATION as above. Then run the script:

```bash
./scripts/multileve_pretrain.sh
```

Around **875min** are needed to pre-train a GPT-2 with multi-level framework. The final loss the model pre-trained with multi levels is 2.700, slightly better than GPT-2 pre-trained from scratch. With multi-level framework, a smaller GPT-2 with 6 layers and 6 heads is trained as the byproduct. Note that the speedup ratio depends on your environment.
