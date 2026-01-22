import pickle
import logging
import os

log_file = "intent_predictions.log"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

model_path = "intent_classifier/classifiers/ml_based/model/intent_classifier_pipeline.pkl"

try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    logging.info(f"Model loaded successfully from: {model_path}")
except Exception as e:
    logging.error(f"Failed to load model from {model_path} | Error: {str(e)}")
    print("Error: Could not load the trained intent model.")
    exit()

def predict_intent(text):
    try:
        if not isinstance(text, str):
            raise ValueError("Input must be a string.")
        if not text.strip():
            raise ValueError("Input is empty.")
        
        intent = model.predict([text])[0]
        logging.info(f"Input: {text} | Predicted Intent: {intent}")
        return intent

    except Exception as e:
        logging.error(f"Prediction failed for input: {text} | Error: {str(e)}")
        return "error"

if __name__ == "__main__":
    print("Type your input (type 'exit' to quit):")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        result = predict_intent(user_input)
        print(f"Predicted Intent: {result}")

