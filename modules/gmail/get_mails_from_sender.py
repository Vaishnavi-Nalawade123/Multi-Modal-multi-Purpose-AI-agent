import base64
from bs4 import BeautifulSoup
from gmail_auth import get_gmail_service

def get_emails_from_sender(sender_email, max_results=10):
    try:
        service = get_gmail_service()
        query = f'from:{sender_email}'
        
     
        response = service.users().messages().list(userId='me', q=query, maxResults=max_results).execute()
        messages = response.get('messages', [])

        if not messages:
            print(f"No messages found from {sender_email}.")
            return []

        email_bodies = []

        for msg in messages:
            msg_detail = service.users().messages().get(userId='me', id=msg['id'], format='full').execute()
            payload = msg_detail.get('payload', {})
            parts = payload.get('parts', [])
            headers = payload.get("headers", [])

            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), "No Subject")

            body = ""

            if 'data' in payload.get('body', {}):
           
                body_data = payload['body']['data']
                decoded_bytes = base64.urlsafe_b64decode(body_data.encode('UTF-8'))
                body = decoded_bytes.decode('utf-8')
            else:
   
                for part in parts:
                    mime_type = part.get('mimeType')
                    body_data = part.get('body', {}).get('data')

                    if body_data:
                        decoded_bytes = base64.urlsafe_b64decode(body_data.encode('UTF-8'))
                        text = decoded_bytes.decode('utf-8')

                        if mime_type == 'text/html':
                            soup = BeautifulSoup(text, 'html.parser')
                            body = soup.get_text()
                            break
                        elif mime_type == 'text/plain' and not body:
                            body = text

            email_bodies.append({
                'subject': subject,
                'body': body.strip()
            })

        return email_bodies

    except Exception as e:
        print(" Error fetching emails:", e)
        return []

if __name__ == "__main__":
    sender = "jaysawant0011@gmail.com"  
    emails = get_emails_from_sender(sender, max_results=5)

    for i, email in enumerate(emails, 1):
        print(f"\n📧 Email {i}")
        print("Subject:", email['subject'])
        print("Body:\n", email['body'])
        print("=" * 60)
