
import os
from datasets import Dataset
from transformers import (GPT2LMHeadModel, GPT2TokenizerFast, DataCollatorForLanguageModeling, Trainer, TrainingArguments)

# Load text data
DATA_FILE = os.path.join('data', 'harry_potter_sample.txt')
with open(DATA_FILE, 'r', encoding='utf-8') as f:
    lines = [line.strip() for line in f if line.strip()]

dataset = Dataset.from_dict({'text': lines})

# Load tokenizer and model
model_name = 'gpt2'
tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(model_name)

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

lm_dataset = dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir='output',
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    logging_steps=10,
    save_steps=50,
    save_total_limit=1,
)

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset,
    data_collator=data_collator,
)

# Train
trainer.train()

# Save model
model.save_pretrained('reinforced_model')
tokenizer.save_pretrained('reinforced_model')

print('Model saved to reinforced_model')


