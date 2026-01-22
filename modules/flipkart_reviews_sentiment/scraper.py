import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.5735.133 Safari/537.36",
]

def get_reviews_from_page(product_url, max_pages=2):
    reviews = []

    for page in range(1, max_pages + 1):
        sep = "&" if "?" in product_url else "?"
        paginated_url = f"{product_url}{sep}page={page}"

        headers = {
            "User-Agent": random.choice(USER_AGENTS)
        }

        print(f"Fetching page {page} with URL: {paginated_url}")

        try:
            response = requests.get(paginated_url, headers=headers)
            print(f"Status code: {response.status_code}")
            
            if response.status_code == 429:
                print(f"Got 429 Too Many Requests on page {page}. Waiting 30 seconds before retry...")
                time.sleep(30)
                continue

            if response.status_code != 200:
                print(f"Failed to fetch page {page}: Status code {response.status_code}")
                break

            soup = BeautifulSoup(response.text, 'html.parser')
            review_blocks = soup.find_all("div", {"class": "_27M-vq"})

            if not review_blocks:
                print(f"No reviews found on page {page}. Stopping.")
                break

            for block in review_blocks:
                review_text_tag = block.find("div", {"class": "t-ZTKy"})
                rating_tag = block.find("div", {"class": "_3LWZlK"})

                review_text = review_text_tag.get_text(strip=True, separator=" ") if review_text_tag else ""
                rating = rating_tag.get_text(strip=True) if rating_tag else ""

                reviews.append({
                    "rating": rating,
                    "review": review_text
                })

            print(f"Scraped page {page} with {len(review_blocks)} reviews.")
            time.sleep(10)

        except Exception as e:
            print(f"Exception occurred on page {page}: {e}")
            break

    return reviews


def main():
    # Replace with your product URL
    product_url = "https://www.flipkart.com/redmi-watch-move-1-85-premium-amoled-14-day-battery-best-accuracy-dual-core-processor-smartwatch/p/itm013acf097d879?pid=SMWHB4YTRWFHFQGY&lid=LSTSMWHB4YTRWFHFQGYP1QQWN&marketplace=FLIPKART&store=ajy%2Fbuh&srno=b_1_1&otracker=browse&fm=organic&iid=en_f33bkWa0ydq3n-YEZ8XRbH_64VT83rXHR7Rkd06ilAGYkgFsN7LMWT0xvRoqujO0_GvpJL9KMIqxg0go8MOa-fUFjCTyOHoHZs-Z5_PS_w0%3D&ppt=hp&ppn=homepage&ssid=0hu66sdnao0000001747841274805"
    reviews = get_reviews_from_page(product_url, max_pages=3)
    
    if not reviews:
        print("No reviews scraped.")
        return

    df = pd.DataFrame(reviews)
    df.to_csv("flipkart_reviews.csv", index=False, encoding='utf-8-sig')
    print(f"Scraped {len(reviews)} reviews and saved to flipkart_reviews.csv")


if __name__ == "__main__":
    main()
