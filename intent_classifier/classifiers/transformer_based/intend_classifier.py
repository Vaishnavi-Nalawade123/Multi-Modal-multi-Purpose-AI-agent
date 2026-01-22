import torch
from transformers import BertForSequenceClassification, BertTokenizer
import joblib
from pathlib import Path

base_dir = Path(__file__).parent / "intent_model"

model = BertForSequenceClassification.from_pretrained(str(base_dir.resolve()))
tokenizer = BertTokenizer.from_pretrained(str(base_dir.resolve()))
label_encoder = joblib.load(str(base_dir / "label_encoder.pkl"))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


def predict_intent(query):
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=64)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        predicted = torch.argmax(outputs.logits, dim=1).item()

    return label_encoder.inverse_transform([predicted])[0]


if __name__ == "__main__":
    while True:
        test_query = input("Enter a query to classify its intent: ")
        predicted_intent = predict_intent(test_query)
        print(f"Predicted intent for query '{test_query}': {predicted_intent}")
    
