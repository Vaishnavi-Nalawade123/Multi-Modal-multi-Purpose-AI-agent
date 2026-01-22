import requests
from bs4 import BeautifulSoup
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from datetime import datetime

finbert_tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
finbert_model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
finbert_labels = ["Positive", "Neutral", "Negative"]

def get_sentiment(text):
    inputs = finbert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = finbert_model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs).item()
    confidence = probs[0][pred].item()
    return finbert_labels[pred], round(confidence, 3)

def analyze_stock_sentiment(url):
    

    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    response.encoding = 'utf-8'
    soup = BeautifulSoup(response.text, 'html.parser')

    articles = soup.find_all('li', class_='clearfix')
    news_data = []

    for article in articles:
        title_tag = article.find('h2')
        link_tag = article.find('a', href=True)

        if title_tag and link_tag:
            title = title_tag.get_text(strip=True)
            link = link_tag['href']
            sentiment, confidence = get_sentiment(title)
            news_data.append({
                'headline': title,
                'link': link,
                'sentiment': sentiment,
                'confidence': confidence
            })

    return pd.DataFrame(news_data)
if __name__ == "__main__":
    
    sentiment_df = analyze_stock_sentiment("https://www.moneycontrol.com/news/tags/tata-motors.html")
    
    print(f"Sentiment analysis for :\n")
    print(sentiment_df)
    
   


