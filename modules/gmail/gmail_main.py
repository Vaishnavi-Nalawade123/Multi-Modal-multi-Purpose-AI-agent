from modules.gmail.sub_intent_classifier.gmail_sub_intent_classifier import predict_sub_intent
from modules.gmail.last_n_mails import get_last_n_emails
from modules.gmail.last_n_mails import extract_count

def gmail_operation(user_input):

    sub_intent = predict_sub_intent(user_input)
    
    if sub_intent == "fetch_last_n":
        n = extract_count(user_input)
        if n is None:
            n = 5
        return get_last_n_emails(n=n) 
    elif sub_intent == "send_email":
        return "Preparing to send an email..."
    elif sub_intent == "fetch_by_sender":
        return "Fetching emails from the specified sender..."
    elif sub_intent == "fetch_unread":
        return "Fetching unread emails..."
    elif sub_intent == "fetch_important":
        return "Fetching important emails..."
    elif sub_intent == "fetch_date_range":
        return "Fetching emails within the specified date range..."
    elif sub_intent == "fetch_by_label":
        return "Fetching emails with the specified label..."
    elif sub_intent == "fetch_by_subject":
        return "Fetching emails with the specified subject..."
    elif sub_intent == "fetch_attachments":
        return "Fetching emails with attachments..."
    else:
        return "Unknown operation. Please try again."
    
if __name__ == "__main__":
    while True:
        if user_input.lower() == "exit":
            print("Exiting Gmail operations.")
            break
        user_input = input("Enter your Gmail operation request: ")
        response = gmail_operation(user_input)
        print(response)