from .vision_ocr import extract_text_from_image
from .gpt_formatter import format_text_with_gpt


def make_notes_from_image(image_path):
    raw_text = extract_text_from_image(image_path)
    formatted_text = format_text_with_gpt(raw_text)
    return formatted_text 

if __name__ == "__main__":
    # Example usage
    image_path = "F:\\ocrtest.jpg"
    notes = make_notes_from_image(image_path)
    print(notes)
