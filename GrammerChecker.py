import nltk
import math
import pandas as pd
from nltk import pos_tag, word_tokenize, ngrams
from collections import Counter
from GrammerRules import *
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sklearn.model_selection import train_test_split
import torch
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class GrammarChecker:
    def __init__(self):
        #Initialize the GrammarChecker with statistical models and deep learning components
        self.bigram_model = Counter()                           #Statistical language model components
        self.trigram_model = Counter()
        self.pos_model = Counter()
        self.total_bigrams = 0
        self.total_trigrams = 0
        self.total_tags = 0
        self.tag_set = set()                                    # Deep learning components (T5 model for grammar correction)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.t5_tokenizer = T5Tokenizer.from_pretrained("vennify/t5-base-grammar-correction")
        self.t5_model = T5ForConditionalGeneration.from_pretrained("vennify/t5-base-grammar-correction").to(self.device)

    # Add-k smoothing 
    def log_prob(self, count, total, vocab_size, k=0.001):
        return math.log((count + k) / (total + k * vocab_size))
    #Calculate log probability with add-k smoothing.
    #  count: Observed count of the n-gram or POS tag
    #  total: Total count of all n-grams or POS tags
    #  vocab_size: Size of the vocabulary
    #  k: Smoothing parameter  
    #Returns:(float)Log probability of the n-gram or POS tag


    def train_models(self, corpus):
        # Load and preprocess training data
        df = pd.read_csv(corpus, header=None, names=["original", "corrected"])
        corrected_sentences = df["corrected"].dropna().tolist()

        # Split into training and validation sets
        train_sentences, val_sentences = train_test_split(
            corrected_sentences, 
            test_size=0.02, 
            random_state=42
        )
        train_text = " ".join(train_sentences).lower()
        tokens = word_tokenize(train_text)                            #Tokenize text
        self.bigram_model = Counter(ngrams(tokens, 2))              #build n-gram models
        self.trigram_model = Counter(ngrams(tokens, 3))
        self.total_bigrams = sum(self.bigram_model.values())
        self.total_trigrams = sum(self.trigram_model.values())

        tagged = pos_tag(tokens)                                    #Build POS tag model     
        self.pos_model = Counter(tag for _, tag in tagged)
        self.total_tags = sum(self.pos_model.values())
        self.tag_set = set(self.pos_model.keys())

    # Grammar correction by deep learning model (Transformer)
    def t5_correct(self, text):
        prompt = f"fix: {text.strip()}"
        inputs = self.t5_tokenizer.encode(prompt, return_tensors="pt", truncation=True).to(self.device)
        outputs = self.t5_model.generate(inputs, max_length=64, num_beams=4, early_stopping=True)
        corrected = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Post-process to remove accidental 'Fix:' if the model outputs it
        if corrected.lower().startswith("fix:"):
            corrected = corrected[4:].strip()

        return corrected

    
    def rule_based_check(self, text):
        # Apply rule-based grammar checking
        errors = []
        sentences = nltk.sent_tokenize(text)
        
        for sentence in sentences:                                  #apply sentence-level grammar rules
            errors.extend(check_subject_verb_agreement(sentence))
            errors.extend(check_article_noun_agreement(sentence))
            errors.extend(check_preposition_usage(sentence))
            errors.extend(check_double_negatives(sentence))
            errors.extend(check_countable_nouns(sentence))
        errors.extend(check_verb_tense_consistency(sentences))      #apply paragraph-level grammar rules

        return errors
    
    def statistical_check(self, text, prob_threshold=-15.0, k=0.001):
        suggestions = []
        errors = []
        combined_errors = []
        error_indices = set()
        tokens = text.split()
        bigrams = list(ngrams(tokens, 2))
        trigrams = list(ngrams(tokens, 3))
        tagged = pos_tag(tokens)

        tag_vocab_size = len(self.tag_set)                          #get vocabulary sizes
        trigram_vocab_size = len(self.trigram_model)
        bigram_vocab_size = len(self.bigram_model)
        flagged = False

        for i, bigram in enumerate(bigrams):                        # Check for unusual bigrams
            prob = self.log_prob(self.bigram_model[bigram], self.total_bigrams, bigram_vocab_size, k)
            if prob < prob_threshold:
                errors.append(f"Unusual bigram: {' '.join(bigram)} (log-prob={prob:.2f})")
                error_indices.update([i, i + 1])
                flagged = True

        for i, trigram in enumerate(trigrams):                      # Trigram errors
            prob = self.log_prob(self.trigram_model[trigram], self.total_trigrams, trigram_vocab_size, k)
            if prob < prob_threshold:
                errors.append(f"Unusual trigram: {' '.join(trigram)} (log-prob={prob:.2f})")
                error_indices.update([i, i + 1, i + 2])
                flagged = True

        for i, (word, tag) in enumerate(tagged):                    # POS tag errors
            tag_prob = self.log_prob(self.pos_model[tag], self.total_tags, tag_vocab_size, k)
            if tag_prob < prob_threshold:
                errors.append(f"Unusual POS tag '{tag}' in word '{word}' (log-prob={tag_prob:.2f})")
                error_indices.add(i)
                flagged = True

        if error_indices:                                           # Combine indices into spans
            sorted_indices = sorted(error_indices)
            span = [sorted_indices[0]]
            for idx in sorted_indices[1:]:
                if idx == span[-1] + 1:
                    span.append(idx)
                else:
                    phrase = " ".join(tokens[i] for i in span)
                    combined_errors.append(phrase)
                    span = [idx]
            if span:                                                # Add last span
                phrase = " ".join(tokens[i] for i in span)
                combined_errors.append(phrase)

        if flagged:                                                 # T5 correction if any error was flagged
            corrected = self.t5_correct(text)
            suggestions.append({
                "error": " / ".join(combined_errors),
                "correction": corrected,
                "type": "grammar",
                "message": "Unusual sentence structure"
            })
            
        for error in errors:
            logging.info(error)

        for suggestion in suggestions:
            logging.info(suggestion)

        return suggestions