from modules.notes_maker.notes_maker import make_notes_from_image
from modules.text_to_audio.text_to_audio import convert_text_to_audio
from intent_classifier.main import classify_intent as predict_intent
# from modules.general_chatting.chat import return_chat
from modules.stock_market_sentiment.stock_sentiment import analyze_stock_sentiment
from modules.stock_market_sentiment.name_extractor import extract_company_name
from modules.gmail.sub_intent_classifier.gmail_sub_intent_classifier import predict_sub_intent
from modules.gmail.gmail_main import gmail_operation

def handle_make_notes():
    image_path = input("Enter the path to the image: ")
    try:
        notes = make_notes_from_image(image_path)
        print("Final Notes:\n", notes)
        convert_text_to_audio(notes, "notes_audio.mp3")
    except Exception as e:
        print(f"Error in making notes: {e}")

# def handle_general_chat(user_input):
#     try:
#         chat_response = return_chat(user_input)
#         print("Chat Response:", chat_response)
#     except Exception as e:
#         print(f"Chat module failed: {e}")

def handle_stock_sentiment(user_input):
    company_name, news_url = extract_company_name(user_input)
    if not company_name or not news_url:
        print("Company not recognized. Please try a valid Indian stock name.")
        return
    try:
        sentiment_df = analyze_stock_sentiment(news_url)
        print(f"\nSentiment analysis for {company_name.upper()}:\n")
        print(sentiment_df.to_string(index=False))
    except Exception as e:
        print(f"Sentiment analysis failed: {e}")

def handle_convert_to_audio():
    text_to_convert = input("Enter the text to convert to audio: ")
    try:
        convert_text_to_audio(text_to_convert, "converted_audio.mp3")
    except Exception as e:
        print(f"Text-to-audio conversion failed: {e}")

def handle_audio_summary():
    file_path = input("Enter the path to the audio file: ")
    try:
        print("Audio summarization not yet implemented.")
    except Exception as e:
        print(f"Audio summarization failed: {e}")

def handle_gmail_operations(user_input):
    try:
        response = gmail_operation(user_input)
        if isinstance(response, list):
            for idx, email in enumerate(response, 1):
                print(f"\nEmail {idx}")
                print("From:", email['From'])
                print("Subject:", email['Subject'])
                print("Date:", email['Date'])
                print("Snippet:", email['Snippet'])
                print("Body:", email['Body'])
        else:
            print(response)
    except Exception as e:
        print(f"Gmail operation failed: {e}")

def main():
    print("AI Agent Initialized. Type 'exit' to quit.")
    while True:
        user_input = input("\n>>> Enter your prompt: ").strip()
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting AI Agent. Goodbye.")
            break
        try:
            intent = predict_intent(user_input) 
            print("Intent Detected:", intent)
        except Exception as e:
            print(f"Intent classification failed: {e}")
            continue

        if intent == "make_notes":
            handle_make_notes()
        # elif intent == "general_chat":
        #     handle_general_chat(user_input)
        elif intent == "flipkart_product_sentiment":
            print("Flipkart sentiment analysis not yet implemented.")
        elif intent == "stock_sentiment":
            handle_stock_sentiment(user_input)
        elif intent == "convert_to_audio":
            handle_convert_to_audio()
        elif intent == "summarize_audio":
            handle_audio_summary()
        elif intent == "gmail_operations":
            handle_gmail_operations(user_input)
        else:
            print("Unknown intent. Try again.")

if __name__ == "__main__":
    main()
