from nltk import pos_tag, word_tokenize, ngrams
from collections import Counter
from GrammerRules import *

# ==============================================
# 01. SUBJECT-VERB AGREEMENT CHECKER
# ==============================================
def check_subject_verb_agreement(sentence):
    errors = []
    words = word_tokenize(sentence)
    tagged = pos_tag(words)

    singular_pronouns = ['he', 'she', 'it']         # Define pronoun categories and modal verbs
    plural_pronouns = ['they', 'we', 'you']
    modals = ['can', 'could', 'will', 'would', 'shall', 'should', 'may', 'might', 'must']

    for i in range(len(tagged) - 1):
        subject, subj_tag = tagged[i]
        verb, verb_tag = tagged[i + 1]
        subject_lc = subject.lower()
        verb_lc = verb.lower()
        # Rule 1: Singular subject needs verb with -s/-ies
        if subj_tag in ['NN', 'NNP'] or subject_lc in singular_pronouns:
            if verb_tag == 'VB' and not verb_lc.startswith('be'):
                correction = verb + 's' if not verb.endswith('y') else verb[:-1] + 'ies'
                errors.append({
                    'error': f"{subject} {verb}",
                    'correction': f"{subject} {correction}",
                    'type': 'grammar',
                    'message': "Singular subject requires -s verb"
                })

        # Rule 2: Plural subject shouldn't have -s on verb
        if subj_tag in ['NNS', 'NNPS'] or subject_lc in plural_pronouns:
            if verb_tag == 'VBZ':
                base_form = verb[:-1] if verb.endswith('s') else verb
                errors.append({
                    'error': f"{subject} {verb}",
                    'correction': f"{subject} {base_form}",
                    'type': 'grammar',
                    'message': "Plural subject takes base-form verb"
                })

        # Rule 3: "I" takes base-form verbs
        if subject_lc == 'I' and verb_tag == 'VBZ':
            base_form = verb[:-1] if verb.endswith('s') else verb
            errors.append({
                'error': f"{subject} {verb}",
                'correction': f"{subject} {base_form}",
                'type': 'grammar',
                'message': "Subject 'I' takes base-form verbs"
            })

        # Rule 4: Handle auxiliary "be" verb cases 
        if verb_lc in ['is', 'are', 'was', 'were']:
            # Singular pronouns need singular "be" forms
            if subject_lc in singular_pronouns + ['i'] and verb_lc in ['are', 'were']:
                correct_form = 'is' if verb_lc == 'are' else 'was'
                errors.append({
                    'error': f"{subject} {verb}",
                    'correction': f"{subject} {correct_form}",
                    'type': 'grammar',
                    'message': f"'{subject}' should be used with '{correct_form}'"
                })
            # Plural pronouns need plural "be" forms
            elif subject_lc in plural_pronouns and verb_lc in ['is', 'was']:
                correct_form = 'are' if verb_lc == 'is' else 'were'
                errors.append({
                    'error': f"{subject} {verb}",
                    'correction': f"{subject} {correct_form}",
                    'type': 'grammar',
                    'message': f"'{subject}' should be used with '{correct_form}'"
                })

    return errors

# ==============================================
# 02. ARTICLE-NOUN AGREEMENT CHECKER
# ==============================================
def check_article_noun_agreement(sentence):
    errors = []
    words = word_tokenize(sentence)
    
    for i in range(len(words)-1):
        current = words[i].lower()
        next_word = words[i+1].lower()
        # Check 'a' before consonant sounds
        if current == 'a' and next_word[0] in 'aeiou':
            # Exceptions for consonant sounds with vowel letters
            if not any(next_word.startswith(ex) for ex in ['uni', 'eu', 'use','user','usual']):
                errors.append({
                    'error': f"{words[i]} {words[i+1]}",
                    'correction': f"an {words[i+1]}",
                    'type': 'grammar',
                    'message': "Use 'an' before vowel sounds"
                })
        
        # Check 'an' before consonant sounds
        elif current == 'an' and next_word[0] not in 'aeiou':
            # Exceptions for vowel sounds with consonant letters
            if not any(next_word.startswith(ex) for ex in ['honest', 'honor', 'hour']):
                errors.append({
                    'error': f"{words[i]} {words[i+1]}",
                    'correction': f"a {words[i+1]}",
                    'type': 'grammar',
                    'message': "Use 'a' before consonant sounds"
                })
        
        # Check for secial cases
        if current == 'a' and next_word[0] not in 'aeiou':
            if any(next_word.startswith(ex) for ex in ['honest', 'honor', 'hour']):
                errors.append({
                    'error': f"{words[i]} {words[i+1]}",
                    'correction': f"an {words[i+1]}",
                    'type': 'grammar',
                    'message': "Use 'a' before consonant sounds"
                })
        if current == 'an' and next_word[0] in 'aeiou':
            if any(next_word.startswith(ex) for ex in ['uni', 'eu', 'one', 'use','user','usual']):
                errors.append({
                    'error': f"{words[i]} {words[i+1]}",
                    'correction': f"an {words[i+1]}",
                    'type': 'grammar',
                    'message': "Use 'a' before vowel sounds"
                })
    return errors

# ==============================================
# 03. VERB TENSE CONSISTENCY CHECKER
# ==============================================
def check_verb_tense_consistency(sentences):
    errors = []
    tense_sequence = []
    
    for sentence in sentences:
        tagged = pos_tag(word_tokenize(sentence))
        verbs = [(word, tag) for word, tag in tagged if tag.startswith('VB')]
        
        if verbs:
            main_verb = verbs[0]
            tense = detect_tense(main_verb)
            tense_sequence.append(tense)
    
    # Check for inconsistent tense usage
    if len(set(tense_sequence)) > 1:
        errors.append({
            'error': "Multiple tenses detected",                    # Flag if multiple tenses detected
            'correction': "Coming Soon",
            'type': 'grammar',
            'message': "Inconsistent verb tense"
        })
    
    return errors

def detect_tense(verb):
    word, tag = verb
    if tag == 'VBD':                    # Past simple
        return 'past'
    elif tag == 'VBG':                  # Present participle/gerund
        return 'continuous'
    elif tag == 'VBN':                  # Past participle
        return 'perfect'
    elif tag == 'VBP' or tag == 'VBZ':  # Present simple
        return 'present'
    return 'unknown'


# ==============================================
# 04. PREPOSITION USAGE CHECKER
# ==============================================
def check_preposition_usage(sentence):
    errors = []
    common_prep_errors = {                       # Dictionary of common preposition errors and their corrections
        'depend of': 'depend on',
        'independent from': 'independent of',
        'similar with': 'similar to',
        'different than': 'different from',
        'comply to': 'comply with',
        'concern on': 'concern about',
        'interested on': 'interested in',
        'listen at': 'listen to',
        'married with': 'married to',
        'discuss about': 'discuss',
        'enter into': 'enter',
        'consist in': 'consist of',
        'accused for': 'accused of',
        'provide with': 'provide',
        'subscribe in': 'subscribe to',
        'agree to': 'agree with',
        'angry to': 'angry with',
        'contribute in': 'contribute to',
        'graduate in': 'graduate from',
        'prevent to': 'prevent from',
        'good in': 'good at',
        'aware about': 'aware of',
        'apply on': 'apply to',
        'object against': 'object to',
        'prefer more than': 'prefer to',
        'rely in': 'rely on',
        'reply back': 'reply',
        'explain about': 'explain',
        'request for': 'request',
        'wait for to': 'wait for',
        'search about': 'search for',
        'emphasize on': 'emphasize',
        'approach to': 'approach',
        'lack in': 'lack',
        'pay for attention': 'pay attention',
        'shout on': 'shout at',
        'aim to': 'aim at',
        'invest on': 'invest in',
        'talk about about': 'talk about',
        'expose with': 'expose to',
        'care about for': 'care for',
        'compare to with': 'compare with',
        'bored from': 'bored with',
        'deal about': 'deal with',
        'engage to': 'engage in',
        'familiar about': 'familiar with'
    }

    words = word_tokenize(sentence)
    split_sentence = sentence.split()  # For getting original capitalization
    
    for i in range(len(words)-1):
        phrase = f"{words[i].lower()} {words[i+1].lower()}"
        if phrase in common_prep_errors:
            if i+1 < len(split_sentence):                                   # Ensure don't go out of bounds
                original = f"{split_sentence[i]} {split_sentence[i+1]}"
                correction = common_prep_errors[phrase]
                
                # Preserve original capitalization in the correction
                if split_sentence[i][0].isupper():
                    correction = correction[0].upper() + correction[1:]
                errors.append({
                    'error': original,
                    'correction': correction,
                    'type': 'grammar',
                    'message': "Common preposition error"
                })
    return errors

# ==============================================
# 05. DOUBLE NEGATIVE CHECKER
# ==============================================
def check_double_negatives(sentence):
    errors = []
    negative_words = ['not', 'never', 'no', 'nobody', 'nothing', 'none', 'neither', 'nowhere']
    words = word_tokenize(sentence.lower())
    
    negatives = [word for word in words if word in negative_words]
    if len(negatives) > 1:
        error_span = " ".join([w for w in sentence.split() if w.lower() in negative_words])
        errors.append({
            'error': error_span,
            'correction': "[Revise to remove double negative]",
            'type': 'grammar',
            'message': "Double negative detected"
        })
    return errors

# ==============================================
# 06. COUNTABLE/UNCOUNTABLE NOUNS CHECKER
# ==============================================
def check_countable_nouns(sentence):
    errors = []

    uncountable_nouns = {                       # Dictionary of uncountable nouns and their correct forms
        'information': 'some information',
        'advice': 'some advice',
        'luggage': 'a piece of luggage',
        'equipment': 'a piece of equipment',
        'furniture': 'a piece of furniture',
        'homework': 'some homework',
        'knowledge': 'some knowledge',
        'money': 'some money',
        'traffic': 'heavy traffic',
        'news': 'some news',
        'progress': 'some progress',
        'research': 'some research',
        'work': 'some work',
        'weather': 'some weather',
        'bread': 'a piece of bread',
        'water': 'some water',
        'music': 'some music',
        'software': 'a software program',
        'meat': 'some meat',
        'sand': 'some sand',
        'rice': 'some rice',
        'paper': 'a piece of paper',
        'furniture': 'a piece of furniture',
        'baggage': 'a piece of baggage',
        'butter': 'some butter',
        'cheese': 'some cheese',
        'coffee': 'a cup of coffee',
        'tea': 'a cup of tea',
        'oil': 'some oil',
        'salt': 'some salt',
        'sugar': 'some sugar',
        'gas': 'some gas',
        'air': 'some air',
        'oxygen': 'some oxygen',
        'carbon': 'some carbon',
        'gold': 'some gold',
        'silver': 'some silver',
        'iron': 'some iron',
        'electricity': 'some electricity',
        'power': 'some power',
        'education': 'some education',
        'literature': 'some literature',
        'beauty': 'some beauty',
        'happiness': 'some happiness',
        'fun': 'some fun',
        'evidence': 'a piece of evidence',
        'damage': 'some damage',
        'clothing': 'an item of clothing',
        'machinery': 'a piece of machinery',
        'vocabulary': 'a set of vocabulary',
        'violence': 'some violence',
        'pollution': 'some pollution',
        'reliability': 'some reliability',
        'plastic': 'some plastic',
        'glass': 'some glass',
        'wood': 'some wood',
        'cotton': 'some cotton',
        'hair': 'a strand of hair',
        'behavior': 'some behavior',
        'equality': 'some equality',
        'justice': 'some justice',
        'luck': 'some luck',
        'news': 'some news',
        'travel': 'some travel',
        'time': 'some time',
        'space': 'some space',
        'noise': 'some noise',
        'rain': 'some rain',
        'snow': 'some snow',
        'ice': 'some ice'
    }
    # Create set of incorrect plural forms
    plural_wrong_forms = {noun + 's' for noun in uncountable_nouns.keys()}

    words = word_tokenize(sentence)
    tagged = pos_tag(words)

    for i in range(len(tagged)-1):
        word, tag = tagged[i]
        next_word, next_tag = tagged[i+1]

        # Rule 1: "a/an" with uncountable noun
        if word.lower() in ['a', 'an'] and next_word.lower() in uncountable_nouns:
            errors.append({
                'error': f"{word} {next_word}",
                'correction': uncountable_nouns[next_word.lower()],
                'type': 'grammar',
                'message': f"'{next_word}' is uncountable and doesn't take '{word}'"
            })

        # Rule 2: "many" with uncountable noun
        if word.lower() == "many" and next_word.lower() in uncountable_nouns:
            errors.append({
                'error': f"{word} {next_word}",
                'correction': uncountable_nouns[next_word.lower()],
                'type': 'grammar',
                'message': f"'{next_word}' is uncountable; use 'much' or proper quantifier"
            })

        # Rule 3: "few" with uncountable noun
        if word.lower() == "few" and next_word.lower() in uncountable_nouns:
            correct_form = "little " + next_word.lower()
            errors.append({
                'error': f"{word} {next_word}",
                'correction': correct_form,
                'type': 'grammar',
                'message': f"'{next_word}' is uncountable; use 'little' instead of 'few'"
            })

        # Rule 4: "much" with countable noun
        if word.lower() == "much" and next_tag in ['NNS']:  # plural form
            errors.append({
                'error': f"{word} {next_word}",
                'correction': f"many {next_word}",
                'type': 'grammar',
                'message': f"'{next_word}' seems countable; use 'many' instead of 'much'"
            })

        # Rule 5: "less" with countable noun
        if word.lower() == "less" and next_tag in ['NNS']:
            errors.append({
                'error': f"{word} {next_word}",
                'correction': f"fewer {next_word}",
                'type': 'grammar',
                'message': f"'{next_word}' is countable; use 'fewer' instead of 'less'"
            })

    for word, tag in tagged:                        # Check for incorrect plural forms of uncountable nouns
        lower_word = word.lower()
        if lower_word in plural_wrong_forms:
            base = lower_word.rstrip('s')
            suggestion = uncountable_nouns.get(base, base)
            errors.append({
                'error': word,
                'correction': suggestion,
                'type': 'grammar',
                'message': f"'{word}' is a plural form of an uncountable noun"
            })

    return errors


