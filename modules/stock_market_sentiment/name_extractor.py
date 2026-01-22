import difflib


stock_name_to_tag_url = {
    "tata motors": "tata-motors",
    "infosys": "infosys",
    "reliance": "reliance-industries",
    "reliance industries": "reliance-industries",
    "hdfc bank": "hdfc-bank",
    "icici bank": "icici-bank",
    "state bank of india": "state-bank-of-india",
    "sbi": "state-bank-of-india",
    "kotak mahindra bank": "kotak-mahindra-bank",
    "mahindra & mahindra": "mahindra-and-mahindra",
    "hdfc": "housing-development-finance-corporation",
    "larsen and toubro": "larsen-and-toubro",
    "axis bank": "axis-bank",
    "bharti airtel": "bharti-airtel",
    "tech mahindra": "tech-mahindra",
    "tcs": "tcs",
    "hindustan unilever": "hindustan-unilever",
    "asian paints": "asian-paints",
    "wipro": "wipro",
    "maruti suzuki": "maruti-suzuki",
    "coal india": "coal-india",
    "bajaj finance": "bajaj-finance",
    "bajaj finserv": "bajaj-finserv",
    "hcl technologies": "hcl-technologies",
    "power grid corporation": "power-grid-corporation",
    "ntpc": "ntpc",
    "nestle india": "nestle-india",
    "sun pharmaceutical": "sun-pharmaceutical-industries",
    "ultratech cement": "ultratech-cement",
    "titan company": "titan-company",
    "indusind bank": "indusind-bank",
    "britannia industries": "britannia-industries",
    "hindalco": "hindalco-industries",
    "jsw steel": "jsw-steel",
    "grasim industries": "grasim-industries",
    "divis laboratories": "divis-laboratories",
    "eicher motors": "eicher-motors",
    "hero motocorp": "hero-motocorp",
    "dr reddys laboratories": "dr-reddys-laboratories",
    "cipla": "cipla",
    "adani enterprises": "adani-enterprises",
    "adani ports": "adani-ports-and-special-economic-zone",
    "adani green energy": "adani-green-energy",
    "adani power": "adani-power",
    "adani total gas": "adani-total-gas",
    "zomato": "zomato",
    "paytm": "paytm",
    "nykaa": "nykaa",
    "delhivery": "delhivery",
    "mrf": "mrf",
    "bosch": "bosch",
    "abb india": "abb-india",
    "page industries": "page-industries",
    "pnb": "punjab-national-bank",
    "canara bank": "canara-bank",
    "bank of baroda": "bank-of-baroda",
    "bandhan bank": "bandhan-bank",
    "idfc first bank": "idfc-first-bank",
    "dlf": "dlf",
    "godrej properties": "godrej-properties",
    "indigo": "interglobe-aviation",
    "gail": "gail-india",
    "bharat petroleum": "bharat-petroleum-corporation",
    "indian oil": "indian-oil-corporation",
    "petronet lng": "petronet-lng"
}


def extract_company_name(prompt, stock_dict = stock_name_to_tag_url, cutoff=0.6):

    prompt_lower = prompt.lower()
    best_match = difflib.get_close_matches(prompt_lower, stock_dict.keys(), n=1, cutoff=cutoff)
    
    if not best_match:
        prompt_tokens = prompt_lower.split()
        for i in range(len(prompt_tokens)):
            for j in range(i+1, len(prompt_tokens)+1):
                phrase = " ".join(prompt_tokens[i:j])
                match = difflib.get_close_matches(phrase, stock_dict.keys(), n=1, cutoff=cutoff)
                if match:
                    best_match = match
                    break
            if best_match:
                break

    if best_match:
        tag = stock_dict[best_match[0]]
        full_url = f"https://www.moneycontrol.com/news/tags/{tag}.html"
        return best_match[0], full_url
    else:
        return None, None


if __name__ == "__main__":
    prompt = input("Enter a stock name or prompt: ")
    company_name, url = extract_company_name(prompt, stock_name_to_tag_url)
    if company_name:
        print(f"Company: {company_name}, URL: {url}")
    else:
        print("No matching company found.")
