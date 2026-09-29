==========
README.txt
==========

Project Title: Writing Correction NLP Tool
-----------------------------------------------------

This project is a comprehensive Natural Language Processing (NLP) application that detects and corrects grammar and spelling errors in English text. It integrates the following features:

1. Rule-based grammar checker
2. Hybrid Statistical grammar checker using n-grams and POS tagging (T5 model-based )
3. SymSpell-based spelling correction

-------------
Directory Structure:
-------------
- main.py                              --> Main appliction to start the writing correction system
- GrammarChecker.py         		      --> Main class for grammar correction using rules, hybrid statistical modeling
- SpellingChecker.py        		      --> Class for spelling error detection and correction using SymSpell
- GrammerRules.py           		      --> Rule-based grammar checking functions (6 rules implemented)
- Cleaned_Lang8.csv         		      --> Corpus: Training data (with corrected and uncorrected sentence pairs)
- frequency_dictionary_en_82_765.txt  	--> Optional frequency dictionary file for SymSpell
- README.txt                		      --> Project overview and usage instructions
- Grammar Model Evaluation folder 	   --> Evaluating the rule-based and statistical model

-------------
Setup Instructions:
-------------
1. Install dependencies:
   pip install nltk / pandas / sklearn / torch / transformers / symspellpy

2. Download NLTK corpora:
   import nltk
   nltk.download('punkt')
   nltk.download('averaged_perceptron_tagger')

3. Make sure Cleaned_Lang8.csv and frequency_dictionary_en_82_765.txt are in the same folder as the code.

4. Run main.py file for the nlp application.

-------------
Implemented Grammar Rules:
-------------
01. Subject-Verb Agreement Checker  
02. Article-Noun Agreement Checker  
03. Verb Tense Consistency Checker (but does not include corrections and grammatical error reason information)  
04. Preposition Usage Checker  
05. Double Negative Checker  
06. Countable/Uncountable Noun Checker  

---------------
Models Applied:
---------------
- Transformer Model: T5 (vennify/t5-base-grammar-correction from HuggingFace)
- Language Modeling: Bigram & Trigram models + POS tag frequency model
- Spelling: SymSpell dictionary lookup (with fallback to custom dictionary from Lang-8 corpus)

-------------
Credits:
-------------
- HuggingFace Transformers
- Lang-8 Learner Corpus
- SymSpell Algorithm (https://github.com/wolfgarbe/SymSpell)
- NLTK Toolkit

==========
README.txt
==========

Project Title: Writing Correction NLP Tool
-----------------------------------------------------

This project is a comprehensive Natural Language Processing (NLP) application that detects and corrects grammar and spelling errors in English text. It integrates the following features:

1. Rule-based grammar checker
2. Hybrid Statistical grammar checker using n-grams and POS tagging (T5 model-based )
3. SymSpell-based spelling correction

-------------
Directory Structure:
-------------
- main.py                              --> Main appliction to start the writing correction system
- GrammarChecker.py         		      --> Main class for grammar correction using rules, hybrid statistical modeling
- SpellingChecker.py        		      --> Class for spelling error detection and correction using SymSpell
- GrammerRules.py           		      --> Rule-based grammar checking functions (6 rules implemented)
- Cleaned_Lang8.csv         		      --> Corpus: Training data (with corrected and uncorrected sentence pairs)
- frequency_dictionary_en_82_765.txt  	--> Optional frequency dictionary file for SymSpell
- README.txt                		      --> Project overview and usage instructions
- Grammar Model Evaluation folder 	   --> Evaluating the rule-based and statistical model

-------------
Setup Instructions:
-------------
1. Install dependencies:
   pip install nltk / pandas / sklearn / torch / transformers / symspellpy

2. Download NLTK corpora:
   import nltk
   nltk.download('punkt')
   nltk.download('averaged_perceptron_tagger')

3. Make sure Cleaned_Lang8.csv and frequency_dictionary_en_82_765.txt are in the same folder as the code.

4. Run main.py file for the nlp application.

-------------
Implemented Grammar Rules:
-------------
01. Subject-Verb Agreement Checker  
02. Article-Noun Agreement Checker  
03. Verb Tense Consistency Checker (but does not include corrections and grammatical error reason information)  
04. Preposition Usage Checker  
05. Double Negative Checker  
06. Countable/Uncountable Noun Checker  

---------------
Models Applied:
---------------
- Transformer Model: T5 (vennify/t5-base-grammar-correction from HuggingFace)
- Language Modeling: Bigram & Trigram models + POS tag frequency model
- Spelling: SymSpell dictionary lookup (with fallback to custom dictionary from Lang-8 corpus)

-------------
Credits:
-------------
- HuggingFace Transformers
- Lang-8 Learner Corpus
- SymSpell Algorithm (https://github.com/wolfgarbe/SymSpell)
- NLTK Toolkit

