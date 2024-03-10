from transformers import AutoTokenizer
from datasets import load_dataset


if __name__ == "__main__":
    num_proc = 32
    test_ratio = 0.05
    max_seq_length = 512
    cache_dir = "cache"

    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", use_fast_tokenizer=True)

    wikipedia_ds = load_dataset(
        "wikipedia",
        "20220301.en",
        cache_dir=cache_dir
    )["train"]

    print(f"Raw Wikipedia-En size: {len(wikipedia_ds)}")

    def tokenize_function(examples, tokenizer, col_name):
        return tokenizer(examples[col_name], return_special_tokens_mask=True)

    # Adapted from https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py
    def group_texts(examples, max_seq_length):
        from itertools import chain
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= max_seq_length:
            total_length = (total_length // max_seq_length) * max_seq_length
        else:
            print("total_length is smaller than block_size")      # avoid a bug in hg
            total_length = 0
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    wikipedia_ds = wikipedia_ds.map(
        tokenize_function,
        num_proc=num_proc,
        fn_kwargs={"tokenizer": tokenizer, "col_name": "text"},
        batched=True,
        remove_columns=wikipedia_ds.column_names,
        desc=f"Tokenize wikipedia",
    ).map(
        group_texts,
        num_proc=num_proc,
        fn_kwargs={"max_seq_length": max_seq_length},
        batched=True,
        desc=f"Group wikipedia",
    )

    wikipedia_ds = wikipedia_ds.train_test_split(test_size=test_ratio)
    print(f"Train Split={len(wikipedia_ds['train'])}, Test Split={len(wikipedia_ds['test'])}")

    wikipedia_ds.save_to_disk(f"datasets/gpt2_wikipedia")

    print("All things done!")
