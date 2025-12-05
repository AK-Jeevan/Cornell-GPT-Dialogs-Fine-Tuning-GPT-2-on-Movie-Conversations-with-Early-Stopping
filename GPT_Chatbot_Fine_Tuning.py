# -----------------------------------------------------------------------------
# GPT Transformer Training with Cornell Movie Dialogs Dataset
#
# This script demonstrates how to:
#   1. Load the Cornell Movie Dialogs dataset
#   2. Tokenize dialogue text with a GPT-2 tokenizer
#   3. Prepare the dataset for causal language modeling
#   4. Fine-tune a GPT-2 model on conversational data with Early Stopping
#   5. Generate sample dialogue responses
# -----------------------------------------------------------------------------

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)

# 1. Load dataset
cornell = load_dataset("cornell_movie_dialog")

# 2. Tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# 3. Tokenize
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_cornell = cornell.map(tokenize_function, batched=True, remove_columns=["text"])

# 4. Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 5. Model
model = GPT2LMHeadModel.from_pretrained("gpt2")

# 6. Training arguments
training_args = TrainingArguments(
    output_dir="./gpt2-cornell-earlystop",
    overwrite_output_dir=True,
    num_train_epochs=10,                
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=100,
    evaluation_strategy="steps",         
    eval_steps=500,                      
    load_best_model_at_end=True,        
    metric_for_best_model="loss",        
    greater_is_better=False              
)

# 7. Trainer with EarlyStoppingCallback
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_cornell["train"],
    eval_dataset=tokenized_cornell["test"],   
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  
)

# 8. Train
trainer.train()

# 9. Test generation
prompt = "Hello, how are you?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(inputs["input_ids"], max_length=50, num_return_sequences=1)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
