from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
from nltk.translate.bleu_score import corpus_bleu
import torch
import numpy as np

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("cssupport/t5-small-awesome-text-to-sql")
model = AutoModelForSeq2SeqLM.from_pretrained("cssupport/t5-small-awesome-text-to-sql")

# Load the dataset (use a suitable evaluation dataset)
dataset = load_dataset("b-mc2/sql-create-context")
print("Available Splits: ", dataset.keys())

def generate_sql(natural_language_query):
    inputs = tokenizer(natural_language_query, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs)
    sql_query = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return sql_query

# Collect references and hypotheses
references = []
hypotheses = []

for example in dataset['train']:
    natural_language_query = example['question']
    reference_sql_query = example['answer']
    generated_sql_query = generate_sql(natural_language_query)
    
    references.append([reference_sql_query.split()])  # BLEU expects tokenized references
    hypotheses.append(generated_sql_query.split())  # BLEU expects tokenized hypotheses

def compute_perplexity(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
        log_likelihood = outputs.loss.item()
    
    perplexity = np.exp(log_likelihood)
    return perplexity

# Example text
text = "How many tables are there"

# Compute perplexity
perplexity = compute_perplexity(model, tokenizer, text)
print(f"Perplexity: {perplexity:.4f}")

# Calculate BLEU score
bleu_score = corpus_bleu(references, hypotheses)
print(f"BLEU Score: {bleu_score:.4f}")
