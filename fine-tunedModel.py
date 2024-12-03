#!/usr/bin/env python
# coding: utf-8

# Import necessary packages
import warnings
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from utils.utils import get_config, print_config, format_dataset, format_string
from utils.LLM import (
    LLM_pretrained,
    LLM_cen_partial,
    get_model,
    get_tokenizer_and_data_collator_and_propt_formatting,
)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Step 1: Load configuration
cfg = get_config("centralized")
print_config(cfg)

# Step 2: Load and preprocess the dataset
trainset_full = load_dataset(cfg.dataset.name, split="train")
train_test = trainset_full.train_test_split(test_size=0.9, seed=1234)
train_dataset = train_test["train"]

train_dataset = format_dataset(train_dataset)
print(f"Number of training examples: {len(train_dataset)}")

# Display an example from the dataset
example_index = 9
data_point = train_dataset[example_index]
print(f"Example Instruction: {data_point['instruction']}")
print(f"Example Response: {data_point['response']}")

# Step 3: Evaluate a pre-trained LLM
llm_pretrained = LLM_pretrained()

prompt = "How to predict the weather"
llm_pretrained.eval(prompt, verbose=False)
llm_pretrained.print_response(verbose=False)

# Evaluate on the dataset example
llm_pretrained.eval(data_point["instruction"], verbose=True)
llm_pretrained.print_response()

# Expected Output
ex_response = format_string(data_point["response"])
print(f"Expected output:\n\t{ex_response}")

# Step 4: Fine-Tuning a Pre-trained LLM
model = get_model(cfg.model)
trainable, all_parameters = model.get_nb_trainable_parameters()
print(f"Trainable parameters: {trainable}")
print(f"All parameters: {all_parameters}")
print(f"Trainable (%): {100 * trainable / all_parameters:.3f}")

# Get tokenizer and data collator
tokenizer, data_collator, format_prompts_fn = get_tokenizer_and_data_collator_and_propt_formatting(
    cfg.model.name, cfg.model.use_fast_tokenizer, cfg.train.padding_side
)

# Define fine-tuning function
save_centralized = "./my_centralized_model"

def finetune_centralised():
    use_cuda = torch.cuda.is_available()
    training_arguments = TrainingArguments(
        **cfg.train.training_arguments,
        use_cpu=not use_cuda,
        output_dir=save_centralized,
    )

    trainer = SFTTrainer(
        tokenizer=tokenizer,
        data_collator=data_collator,
        formatting_func=format_prompts_fn,
        max_seq_length=cfg.train.seq_length,
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
    )

    trainer.train()
    model.save_pretrained(save_centralized)

# Perform fine-tuning
finetune_centralised()

# Step 5: Evaluate the Fine-Tuned Model
llm_cen = LLM_cen_partial()
llm_cen.eval(data_point["instruction"], verbose=True)
llm_cen.print_response()

ex_response = format_string(data_point["response"])
print(f"Expected output:\n\t{ex_response}")

# Step 6: Visualize Results
# Assuming visualize_results is defined in utils
from utils.utils import visualize_results
visualize_results(results=["7b/pretrained", "7b/cen_10"])
