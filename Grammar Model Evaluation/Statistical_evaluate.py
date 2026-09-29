import sys
import os

# Add the parent directory (project_root) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from GrammerChecker import *
from tqdm import tqdm

# Load dataset and split into training and validation
df = pd.read_csv("Cleaned_Lang8.csv", header=None, names=["original", "corrected"])
df = df.dropna()

# Same split as training (must use same random_state!)
_, val_sentences = train_test_split(
    df, test_size=0.02, random_state=42
)

# Initialize grammar checker and train on training data only
checker = GrammarChecker()
checker.train_models("Cleaned_Lang8.csv")  # This trains only on 80%, not validation

# Evaluate T5 correction on validation set
total = len(val_sentences)
total_bleu = 0
correct_count = 0

print("Evaluating on validation set...")
for i, row in tqdm(val_sentences.iterrows(), total=total):
    original = row["original"]
    target = row["corrected"]

    # Get model prediction
    predicted = checker.t5_correct(original)

    # BLEU score (for sentence similarity)
    reference = [target.split()]
    hypothesis = predicted.split()
    smoothing = SmoothingFunction().method1
    bleu = sentence_bleu(reference, hypothesis, smoothing_function=smoothing)
    total_bleu += bleu

    # Exact match
    if predicted.strip().lower() == target.strip().lower():
        correct_count += 1

# Average BLEU and Accuracy
average_bleu = total_bleu / total
accuracy = correct_count / total

print(f"\nHybrid Statistical Model Evaluation Results on Validation Set:")
print(f"Average BLEU Score: {average_bleu:.4f}")
print(f"Exact Match Accuracy: {accuracy:.4f}")

