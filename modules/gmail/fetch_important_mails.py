from gmail_auth import get_gmail_service
import base64
import html
from bs4 import BeautifulSoup

def get_important_emails_with_content(max_results=5):
    service = get_gmail_service()
    query = "is:important"

    response = service.users().messages().list(userId='me', q=query, maxResults=max_results).execute()
    messages = response.get('messages', [])

    email_details = []

    for msg in messages:
        msg_detail = service.users().messages().get(userId='me', id=msg['id'], format='full').execute()

        payload = msg_detail.get('payload', {})
        headers = payload.get('headers', [])
        
        email_info = {
            'From': '',
            'Subject': '',
            'Date': '',
            'Snippet': msg_detail.get('snippet', ''),
            'Body': ''
        }

        for header in headers:
            if header['name'] == 'From':
                email_info['From'] = header['value']
            elif header['name'] == 'Subject':
                email_info['Subject'] = header['value']
            elif header['name'] == 'Date':
                email_info['Date'] = header['value']

    
        body = ''
        parts = payload.get('parts', [])

        for part in parts:
            mime_type = part.get('mimeType')
            body_data = part.get('body', {}).get('data')

            if body_data:
                decoded_bytes = base64.urlsafe_b64decode(body_data.encode('UTF-8'))
                text = decoded_bytes.decode('utf-8')

                if mime_type == 'text/html':
                    # Clean HTML to plain text
                    soup = BeautifulSoup(text, 'html.parser')
                    body = soup.get_text()
                    break
                elif mime_type == 'text/plain' and not body:
                    body = text

        email_info['Body'] = body.strip()
        email_details.append(email_info)

    return email_details

if __name__ == "__main__":
    emails = get_important_emails_with_content()

    for idx, email in enumerate(emails, 1):
        print(f"\n Email {idx}")
        print("From:", email['From'])
        print("Subject:", email['Subject'])
        print("Date:", email['Date'])
        print("Snippet:", email['Snippet'])
        print("Body Preview:", email['Body'][:300], "...\n")
