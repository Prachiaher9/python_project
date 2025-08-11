
# backend/train.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Hugging Face ka pretrained multilingual sentiment model
model_name = "tabularisai/multilingual-sentiment-analysis"

# Tokenizer aur Model load kar rahe hain
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Model ko local folder me save kar rahe hain taki API me use ho sake
save_path = "model/sentiment_xlm"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"✅ Model downloaded and saved to {save_path}")
