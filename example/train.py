import os

from dataclasses import dataclass, field
from typing import Optional
from transformers import (
    Trainer, 
    TrainingArguments, 
    HfArgumentParser, 
    default_data_collator,
    set_seed
)
from transformers.models.gpt2 import (
    GPT2Config,
    GPT2LMHeadModel
)
from datasets import load_from_disk


@dataclass
class ModelArguments:
    config_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "config file for the model. Don't set if you want to train a model from a pretrained one."
        }
    )
    model_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "model path to initialized the model."
        }
    )


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)

    if not os.path.exists("datasets/gpt2_wikipedia"):
        print("Dataset does not exist. Please run data.py at first.")
    dataset = load_from_disk("datasets/gpt2_wikipedia")

    if model_args.config_path:
        config = GPT2Config.from_json_file(model_args.config_path)
        model = GPT2LMHeadModel(config)
        print("Initiating a new model and training from scratch.")
    elif model_args.model_path:
        model = GPT2LMHeadModel.from_pretrained(model_args.model_path)
        print("Initiating a model from a pretrained one.")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        data_collator=default_data_collator
    )

    train_result = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
