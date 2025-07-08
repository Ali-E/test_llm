import os
import random
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# Models to load
MODEL_DIRS = ["gpt2", "reinforced_model"]

# Load some sample questions from the fine-tuning data
DATA_FILE = os.path.join("data", "harry_potter_sample.txt")
with open(DATA_FILE, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip() and "?" in line]

# pick a few questions from the data
random.seed(0)
questions_from_data = random.sample(lines, k=min(3, len(lines)))

# questions unrelated to Harry Potter
other_questions = [
    "What is the capital of France?",
    "Who wrote 'Pride and Prejudice'?",
    "What is the speed of light?",
]

questions = questions_from_data + other_questions

for model_dir in MODEL_DIRS:
    print(f"\nLoading model: {model_dir}")
    try:
        tokenizer = GPT2TokenizerFast.from_pretrained(model_dir)
        model = GPT2LMHeadModel.from_pretrained(model_dir)
    except Exception as e:
        print(f"Failed to load {model_dir}: {e}")
        continue

    for q in questions:
        inputs = tokenizer.encode(q, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(inputs, max_length=50, num_return_sequences=1)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nPrompt: {q}\nAnswer: {answer}\n")
