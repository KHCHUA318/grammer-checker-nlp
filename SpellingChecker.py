import re
import os
import pandas as pd
from collections import defaultdict
from symspellpy import SymSpell, Verbosity

class SpellingChecker:
    def __init__(self):
        self.sym_spell = initialize_symspell()
    
    def check_spelling(self, text):
        suggestions = []            
        words = re.findall(r'\b\w+\b', text)                #Extract words while preserving original form
        
        for word in words:
            #Clean word for dictionary lookup (lowercase, no punctuation)
            clean_word = re.sub(r'[^\w]', '', word.lower())
            if clean_word:
                #Get spelling suggestions
                suggestions_for_word = self.sym_spell.lookup(clean_word, Verbosity.TOP, max_edit_distance=2)
                #If top suggestion differs from original, it's likely an error
                if suggestions_for_word and suggestions_for_word[0].term != clean_word:
                    original = word
                    correction = suggestions_for_word[0].term
                    if word[0].isupper():                   # Preserve original capitalization
                        correction = correction.capitalize()
                    suggestions.append({
                        'error': original,
                        'correction': correction,
                        'type': 'spelling',
                        'message': f"Possible spelling error for '{original}'"
                    })
        return suggestions
        
def load_lang8_corpus(file_path):
    try:
        df = pd.read_csv(file_path, header=None, sep='\t')
        original_sentences = df[0].tolist()
        corrected_sentences = df[1].tolist()
        return original_sentences, corrected_sentences
        if df.empty:
            raise ValueError("Loaded CSV file is empty")
    except Exception as e:
        print(f"Error loading corpus: {e}")
        return [], []               # Return empty lists on failure

def initialize_symspell():
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load a frequency dictionary
    dictionary_path = os.path.join(script_dir, "frequency_dictionary_en_82_765.txt")
    if os.path.exists(dictionary_path):
        sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
    else:
    # Create a simple dictionary from the corpus
        original, corrected = load_lang8_corpus(os.path.join(script_dir, "Cleaned_Lang8.csv"))
        word_freq = defaultdict(int)
        for sentence in corrected:
            for word in sentence.split():
                clean_word = re.sub(r'[^\w]', '', word.lower())
                if clean_word:
                    word_freq[clean_word] += 1
        
        # Add words to SymSpell
        for word, freq in word_freq.items():
            sym_spell.create_dictionary_entry(word, freq)
    
    return sym_spell