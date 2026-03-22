import re
import collections

# -------------------------------------------------
# Sample corpus (replace with large dataset for real use)
# -------------------------------------------------
corpus = """
machine learning is fun and powerful. machine learning algorithms are used everywhere.
deep learning is a subset of machine learning. spelling mistakes can be corrected.
"""

# -------------------------------------------------
# Tokenization
# -------------------------------------------------
def words(text):
    return re.findall(r'\w+', text.lower())

# Build frequency model
word_counts = collections.Counter(words(corpus))
total_words = sum(word_counts.values())

# Probability model P(word)
def P(word):
    return word_counts[word] / total_words if word in word_counts else 0

WORDS = set(word_counts.keys())

# -------------------------------------------------
# Edit Distance Functions
# -------------------------------------------------
def edits1(word):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    
    return set(deletes + transposes + replaces + inserts)

def edits2(word):
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

# -------------------------------------------------
# Known words
# -------------------------------------------------
def known(words):
    return set(w for w in words if w in WORDS)

# -------------------------------------------------
# Correction function
# -------------------------------------------------
def candidates(word):
    return (known([word]) or
            known(edits1(word)) or
            known(edits2(word)) or
            [word])

def correct(word):
    return max(candidates(word), key=P)

# -------------------------------------------------
# Test
# -------------------------------------------------
misspelled = ["machin", "lerning", "deeep", "corected", "algoritms"]

for w in misspelled:
    print(f"{w} → {correct(w)}")