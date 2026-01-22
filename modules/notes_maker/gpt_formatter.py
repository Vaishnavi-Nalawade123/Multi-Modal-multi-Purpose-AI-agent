import ollama

def format_text_with_gpt(raw_text, model="llama3.1"):
    prompt = f"Please convert the following raw extracted text into clean, readable notes with bullet points if needed:\n\n{raw_text}"

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response['message']['content'].strip()
    except Exception as e:
        return f"Error generating notes: {e}"

