import os.path
import pickle
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def get_gmail_service():
    creds = None

    if os.path.exists('token.json'):
        with open('token.json', 'rb') as token:
            creds = pickle.load(token)


    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:

            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials/client_secret_13233244404-4ld1lfri2r07gplv95gna1j2mk1r4bg5.apps.googleusercontent.com.json', SCOPES)
            creds = flow.run_local_server(port=0)

        with open('token.json', 'wb') as token:
            pickle.dump(creds, token)


    service = build('gmail', 'v1', credentials=creds)
    return service
if __name__ == "__main__":
    service = get_gmail_service()
    print("Gmail service created successfully!")
