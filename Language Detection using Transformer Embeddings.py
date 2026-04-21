from sentence_transformers import SentenceTransformer, util
 
# Load multilingual embedding model (LaBSE)
model = SentenceTransformer("sentence-transformers/LaBSE")
 
# Prototype sentences for each language
language_examples = {
    "English": "This is a sample sentence.",
    "Spanish": "Esta es una frase de ejemplo.",
    "French": "Ceci est une phrase d'exemple.",
    "German": "Dies ist ein Beispielsatz.",
    "Italian": "Questa è una frase di esempio.",
    "Portuguese": "Esta é uma frase de exemplo.",
    "Hindi": "यह एक उदाहरण वाक्य है।",
    "Japanese": "これは例文です。"
}
 
# Encode language prototypes
langs = list(language_examples.keys())
prototypes = list(language_examples.values())
prototype_embeddings = model.encode(prototypes, convert_to_tensor=True)
 
# Input text for detection
text_input = "Dove posso trovare una farmacia vicino a me?"
 
# Encode input sentence
input_embedding = model.encode(text_input, convert_to_tensor=True)
 
# Compute cosine similarity to each language
similarities = util.pytorch_cos_sim(input_embedding, prototype_embeddings)[0]
best_lang_index = int(similarities.argmax())
predicted_language = langs[best_lang_index]
 
# Display result
print("🌐 Input Text:\n", text_input)
print("\n🔤 Detected Language:", predicted_language)
print("\n🔢 Similarity Score: {:.2f}".format(similarities[best_lang_index].item()))