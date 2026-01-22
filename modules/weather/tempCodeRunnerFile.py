import requests

API_KEY = "671ec5dca70e76a538e9b9d3fc36182d"  # Replace with your actual API key
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

def get_weather(city_name="Pune"):
    params = {
        "q": city_name,
        "appid": API_KEY,
        "units": "metric"  # Celsius units
    }
    
    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()  # Raises HTTPError if status != 200
        
        data = response.json()
        
        weather_description = data['weather'][0]['description'].capitalize()
        temp = data['main']['temp']
        feels_like = data['main']['feels_like']
        humidity = data['main']['humidity']
        wind_speed = data['wind']['speed']
        
        weather_report = (
            f"Weather in {city_name}:\n"
            f"{weather_description}\n"
            f"Temperature: {temp}°C (Feels like {feels_like}°C)\n"
            f"Humidity: {humidity}%\n"
            f"Wind Speed: {wind_speed} m/s"
        )
        return weather_report
    
    except requests.exceptions.RequestException as e:
        return f"Failed to get weather data: {e}"

if __name__ == "__main__":
    city = input("Enter city name (default is Pune): ") or "Pune"
    print(get_weather(city))
