import re
import logging


logging.basicConfig(
    filename="intent_classifier.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def classify_intent(user_input):
    try:
        
        if not isinstance(user_input, str):
            raise ValueError("Input must be a string.")
        if not user_input.strip():
            raise ValueError("Input string is empty.")

        text = user_input.lower()
        
        if re.search(r"(convert|change).*(text).*(audio|speech)", text):
            intent = "convert_text_to_audio"
        
        elif re.search(r"(stock|share|sentiment|bullish|bearish|price)", text):
            intent = "stock_sentiment"
        
        elif re.search(r"(summarize|summary|make notes|key points|condense)", text):
            intent = "notes_summary"
        
        elif re.search(r"(weather|temperature|forecast|rain|climate)", text):
            intent = "weather"
        
        elif re.search(r"(flipkart|product|reviews|rating|feedback)", text):
            intent = "flipkart_reviews"
        
        elif re.search(r"(sql|database|table|query|select|insert|delete|update|columns|rows|fetch|data)", text):
            intent = "nl2sql"
        
        else:
            intent = "unknown"

        logging.info(f"Input: {user_input} | Intent: {intent}")
        return intent

    except Exception as e:
        logging.error(f"Error processing input: {user_input} | Error: {str(e)}")
        return "error"

if __name__ == "__main__":
    while True:
        query = input("Enter your query (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        intent = classify_intent(query)
        print(f"Input: {query} --> Intent: {intent}")
