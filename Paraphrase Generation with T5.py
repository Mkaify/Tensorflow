from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
import tensorflow as tf

# Better paraphrasing model
model_name = "ramsrigouthamg/t5_paraphraser"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)

sentence = "The weather today is beautiful with clear skies and sunshine."

# Proper paraphrasing prompt
text = "paraphrase: " + sentence + " </s>"

encoding = tokenizer.encode_plus(
    text,
    padding="max_length",
    max_length=128,
    return_tensors="tf",
    truncation=True
)

input_ids = encoding["input_ids"]

outputs = model.generate(
    input_ids,
    max_length=64,
    num_beams=5,
    num_return_sequences=1,
    early_stopping=True
)

paraphrase = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Original:", sentence)
print("Paraphrase:", paraphrase)