import os
import io
from google.cloud import vision


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "modules/notes_maker/diesel-ability-458914-q4-c7d4e19a47bb.json"


def extract_text_from_image(image_path):
    client = vision.ImageAnnotatorClient()

    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.document_text_detection(image=image)

    if response.error.message:
        raise Exception(f"API Error: {response.error.message}")
    
    return response.full_text_annotation.text

if __name__ == "__main__":
    image_path =  "modules/notes_maker/sample_image.jpg"
    try:
        text = extract_text_from_image(image_path)
        print("Extracted Text:")
        print(text)
    except Exception as e:
        print(f"Error: {e}")
    
