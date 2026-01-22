from faster_whisper import WhisperModel
import warnings
warnings.filterwarnings("ignore")

def transcribe_audio_file(file_path, model_size="base"):
    model = WhisperModel(model_size)
    segments, info = model.transcribe(file_path)
    full_text = " ".join([segment.text for segment in segments])
    return full_text

if __name__ == "__main__":
    file_path = "C:\\Users\\91966\\Desktop\\Recording.m4a"
    transcript = transcribe_audio_file(file_path)
    print("Transcribed text:", transcript)
