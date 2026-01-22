from .voice_transcribe import transcribe_audio_file
# gpt_formatter.py

import ollama

def summarize_transcribe(transcribed_text, model="llama3.1"):
    prompt = f"please summarize this audio transcript provided below:\n\n{transcribed_text}\n\nPlease provide a concise summary of the main points and key details."
    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response['message']['content'].strip()
    except Exception as e:
        return f"⚠️ Error generating notes: {e}"
