import requests
import spacy

API_KEY = "671ec5dca70e76a538e9b9d3fc36182d"  
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

nlp = spacy.load("en_core_web_sm")

def extract_city(prompt):
    prompt = prompt.title()  # Capitalize first letter of each word
    doc = nlp(prompt)
    for ent in doc.ents:
        if ent.label_ == "GPE":
            return ent.text
    return "Pune"

def get_weather(prompt):
    city_name = extract_city(prompt)
    params = {
        "q": city_name,
        "appid": API_KEY,
        "units": "metric"  
    }
    
    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()  
        
        data = response.json()
        
        weather_description = data['weather'][0]['description'].capitalize()
        temp = data['main']['temp']
        feels_like = data['main']['feels_like']
        humidity = data['main']['humidity']
        wind_speed = data['wind']['speed']
        
        weather_report = (
            f"🌤️ **Weather Update for {city_name}**\n\n"
            f"- **Condition:** {weather_description}\n"
            f"- **Temperature:** {temp} °C\n"
            f"- **Feels Like:** {feels_like} °C\n"
            f"- **Humidity:** {humidity}%\n"
            f"- **Wind Speed:** {wind_speed} m/s"
        )

        return weather_report
    
    except requests.exceptions.RequestException as e:
        return f"Failed to get weather data: {e}"

if __name__ == "__main__":
    prompt = input("Ask about the weather: ").strip()
    print(get_weather(prompt))
