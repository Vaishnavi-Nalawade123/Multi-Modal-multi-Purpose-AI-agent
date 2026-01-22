import pyttsx3


def convert_text_to_audio(text, filename="notes_audio.mp3"):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Speed
    engine.setProperty('volume', 1.0)  # Volume
    engine.save_to_file(text, filename)
    engine.runAndWait()
    print(f"🔊 Audio saved as {filename}")
