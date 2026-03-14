import spacy
 
# Load spaCy English pipeline with transformer-based NER (small model for demo)
# For better results, use: en_core_web_trf (requires: python -m spacy download en_core_web_trf)
nlp = spacy.load("en_core_web_sm")
 
# Input text
text = "Apple is opening a new office in Toronto on January 15th, 2025."
 
# Run NER
doc = nlp(text)
 
# Extract and print named entities
print("📄 Text:\n", text)
print("\n🔍 Detected Named Entities:\n")
for ent in doc.ents:
    print(f"{ent.text:30} → {ent.label_}")