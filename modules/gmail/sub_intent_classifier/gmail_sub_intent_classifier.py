import joblib

vectorizer = joblib.load("modules/gmail/sub_intent_classifier/model/tfidf_vectorizer.pkl")
model = joblib.load("modules/gmail/sub_intent_classifier/model/sub_intent_model.pkl")

def predict_sub_intent(user_input):
    X = vectorizer.transform([user_input])
    predicted_intent = model.predict(X)[0]
    return predicted_intent

if __name__ == "__main__":
    sample = "Get pervious emails from jay"
    print(f"Predicted sub-intent: {predict_sub_intent(sample)}")
