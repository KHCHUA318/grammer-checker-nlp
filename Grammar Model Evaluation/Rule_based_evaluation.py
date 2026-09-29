import sys
import os
# Add the parent directory (project_root) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from GrammerChecker import *
from tqdm import tqdm
import nltk

# Ensure NLTK models are downloaded
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")

# Load dataset and split into training and validation
df = pd.read_csv("Cleaned_Lang8.csv", header=None, names=["original", "corrected"])
df = df.dropna()

# Use the same split as training to avoid data leakage
_, val_sentences = train_test_split(df, test_size=0.02, random_state=42)

# Initialize grammar checker and train the model
checker = GrammarChecker()
checker.train_models("Cleaned_Lang8.csv")  # trains on 98%, val set is held-out

# Helper: Apply rule-based corrections
def apply_rule_based_corrections(text, errors):
    tokens = nltk.word_tokenize(text)
    for err in errors:
        incorrect = err['error']
        correction = err['correction']
        incorrect_tokens = nltk.word_tokenize(incorrect)
        correction_tokens = nltk.word_tokenize(correction)

        for i in range(len(tokens) - len(incorrect_tokens) + 1):
            if tokens[i:i + len(incorrect_tokens)] == incorrect_tokens:
                tokens = tokens[:i] + correction_tokens + tokens[i + len(incorrect_tokens):]
                break
    return " ".join(tokens)

# Evaluation
total = len(val_sentences)
total_bleu = 0
correct_count = 0

print("Evaluating rule-based model on validation set...")
for i, row in tqdm(val_sentences.iterrows(), total=total):
    original = row["original"]
    target = row["corrected"]

    # Run rule-based checker
    errors = checker.rule_based_check(original)
    if errors:
        predicted = apply_rule_based_corrections(original, errors)
    else:
        predicted = original

    # BLEU Score
    reference = [target.split()]
    hypothesis = predicted.split()
    smoothing = SmoothingFunction().method1
    bleu = sentence_bleu(reference, hypothesis, smoothing_function=smoothing)
    total_bleu += bleu

    # Exact match
    if predicted.strip().lower() == target.strip().lower():
        correct_count += 1

# Final metrics
average_bleu = total_bleu / total
accuracy = correct_count / total

print(f"\nRule-Based Model Evaluation Results on Validation Set:")
print(f"Average BLEU Score: {average_bleu:.4f}")
print(f"Exact Match Accuracy: {accuracy:.4f}")
