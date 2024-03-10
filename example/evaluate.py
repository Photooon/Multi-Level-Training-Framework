import os
os.environ["WANDB_DISABLED"] = "true"

from argparse import ArgumentParser
from transformers import (
    Trainer, 
    default_data_collator,
)
from transformers.models.gpt2 import (
    GPT2LMHeadModel
)
from datasets import load_from_disk

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None, required=True)
    args = parser.parse_args()

    if not os.path.exists("datasets/gpt2_wikipedia"):
        print("Dataset does not exist. Please run data.py at first.")
    dataset = load_from_disk("datasets/gpt2_wikipedia")

    model = GPT2LMHeadModel.from_pretrained(args.model_path)

    trainer = Trainer(
        model=model,
        eval_dataset=dataset["test"],
        data_collator=default_data_collator
    )

    eval_result = trainer.evaluate()
    trainer.log_metrics("test", eval_result)
