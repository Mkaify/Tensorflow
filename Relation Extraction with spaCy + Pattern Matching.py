import spacy
from spacy.matcher import Matcher
 
# Load spaCy language model
nlp = spacy.load("en_core_web_sm")
 
# Input sentence
text = "Barack Obama was the president of the United States and worked for the government."
 
# Define matcher for "PERSON works for ORG"
matcher = Matcher(nlp.vocab)
pattern = [
    {"ENT_TYPE": "PERSON"},
    {"LEMMA": "work", "OP": "?"},
    {"LOWER": "for"},
    {"ENT_TYPE": "ORG"}
]
matcher.add("WORKS_FOR", [pattern])
 
# Process the text
doc = nlp(text)
 
# Find matches
matches = matcher(doc)
print("📄 Text:\n", text)
print("\n🔗 Extracted Relations:\n")
for match_id, start, end in matches:
    span = doc[start:end]
    person = [ent.text for ent in span.ents if ent.label_ == "PERSON"]
    org = [ent.text for ent in span.ents if ent.label_ == "ORG"]
    if person and org:
        print(f"{person[0]} → works for → {org[0]}")