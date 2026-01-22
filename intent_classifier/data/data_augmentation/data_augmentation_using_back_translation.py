from deep_translator import GoogleTranslator
import time
import pandas as pd

data1 = pd.read_csv("raw\\gmail_intent_data.csv")

augmented = []

for idx, row in data1.iterrows():
    try:
        fr = GoogleTranslator(source='en', target='fr').translate(row["text"])
        en = GoogleTranslator(source='fr', target='en').translate(fr)
        
        augmented.append({"text": en, "intent": row["intent"]})
        
        time.sleep(1)  # to avoid API rate limits
    except Exception as e:
        print(f"Error translating: {row['text']} → {e}")

augmented_df = pd.DataFrame(augmented)
augmented_df.to_csv("gmail_augmented_backtranslated.csv", index=False)

print(augmented_df.head())
