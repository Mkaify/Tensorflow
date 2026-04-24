import spacy
import coreferee

# Load spaCy model and add Coreferee pipeline
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe('coreferee')

# Input text with pronouns
text = "Angela went to the market. She bought some apples. Then she met her friend."

# Process the text
doc = nlp(text)

# Display results
print("📄 Original Text:\n", text)
print("\n🔗 Coreference Clusters:\n")
for chain in doc._.coref_chains:
    mentions = [doc[span.start:span.end].text for span in chain]
    print(" → ".join(mentions))