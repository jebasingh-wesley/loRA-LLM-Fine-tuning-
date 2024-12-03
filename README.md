# **Centralized LLM Fine-Tuning with LoRA**

This repository contains a step-by-step guide to fine-tuning Large Language Models (LLMs) using Low-Rank Adaptation (LoRA). The goal is to train and evaluate domain-specific LLMs with a centralized approach, leveraging the robust **ChatGLM2-6B** base model and medAlpaca dataset for practical examples.

---

## **Table of Contents**
1. [Overview](#overview)  
2. [Key Features](#key-features)  
3. [Requirements](#requirements)  
4. [Installation](#installation)  
5. [How to Run](#how-to-run)  
6. [Evaluation](#evaluation)  
7. [Results Visualization](#results-visualization)  
8. [License](#license)  

---

## **Overview**
This project demonstrates:
- Fine-tuning LLMs using **LoRA** for domain-specific tasks.  
- Centralized training on the medAlpaca dataset, which is specifically curated for medical Q&A tasks.  
- Comparing the performance of pre-trained models and fine-tuned models.  

The process includes loading a dataset, configuring a model, performing centralized fine-tuning, and evaluating the model's performance.

---

## **Key Features**
- **Pre-trained LLM Setup**: Use the pre-trained **ChatGLM2-6B** for domain-specific queries.  
- **Dataset Handling**: Load and preprocess datasets like **medAlpaca**.  
- **Fine-Tuning with LoRA**: Efficiently adapt large models to new tasks without requiring extensive computational resources.  
- **Systematic Evaluation**: Visualize and compare results between pre-trained and fine-tuned LLMs.  

---

## **Requirements**
Ensure you have the following dependencies installed:
- Python 3.8+
- PyTorch
- Transformers
- TRL (🤗 `trl` library for fine-tuning)
- Datasets
- Hydra
- Matplotlib (for visualization)

---

## **Installation**
Clone this repository and install the required dependencies:
```bash
git clone https://github.com/yourusername/centralized-llm-finetuning.git
cd centralized-llm-finetuning

# Install dependencies
pip install -r requirements.txt
```

---

## **How to Run**

### Step 1: Load Configurations  
Edit and load the centralized configuration:
```python
cfg = get_config("centralized")
print_config(cfg)
```

### Step 2: Load and Prepare the Dataset  
The dataset is split into training and testing sets, formatted for fine-tuning:
```python
trainset_full = load_dataset(cfg.dataset.name, split='train')
train_dataset = format_dataset(train_test["train"])
```

### Step 3: Run Fine-Tuning  
Use the `finetune_centralised` function to fine-tune the base model:
```python
finetune_centralised()
```

### Step 4: Evaluate the Model  
Evaluate the pre-trained and fine-tuned models:
```python
llm_cen.eval(data_point['instruction'], verbose=True)
```

---

## **Evaluation**
Run systematic evaluation for accuracy:
```python
# Generate answers for the pre-trained model
inference(base_model_name_path=cfg.model.name, run_name="pretrained")

# Generate answers for the fine-tuned model
inference(
    base_model_name_path=cfg.model.name,
    peft_path="./my_centralized_model",
    run_name="centralised_finetuned"
)

# Evaluate results
evaluate(run_name="pretrained")
evaluate(run_name="centralised_finetuned")
```

---

## **Results Visualization**
Visualize the difference in performance:
```python
visualize_results(results=['7b/pretrained', '7b/cen_10'])
```

---

## **File Structure**
```plaintext
centralized-llm-finetuning/
├── utils/                     # Utility functions and scripts
│   ├── LLM.py                 # LLM setup and configuration
│   ├── utils.py               # Helper functions for dataset and evaluation
├── configs/                   # Configuration files for training and evaluation
├── requirements.txt           # Python dependencies
├── centralized_finetune.ipynb # Main notebook for centralized fine-tuning
└── README.md                  # Project documentation
```

---

## **License**
This project is licensed under the MIT License. See the `LICENSE` file for details.

