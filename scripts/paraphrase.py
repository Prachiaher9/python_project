# backend/scripts/paraphrase.py
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
model_name = "ramsrigouthamg/t5_paraphraser"  # example

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def paraphrase_text(text, n=3):
    input_text = "paraphrase: " + text + " </s>"
    encoded = tokenizer.encode_plus(input_text, return_tensors="pt", truncation=True)
    outputs = model.generate(**encoded, max_length=128, num_return_sequences=n, do_sample=True, top_k=50)
    result = [tokenizer.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True) for out in outputs]
    return list(dict.fromkeys(result))  # unique
