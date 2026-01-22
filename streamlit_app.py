# ------------------------ Import necessary libraries -------------------
import streamlit as st
import logging
import tempfile
import os
import base64
import requests
import spacy
import matplotlib.pyplot as plt
import pandas as pd

# ------------------------ Internal Modules -----------------------------
from modules.notes_maker.notes_maker import make_notes_from_image
from modules.text_to_audio.text_to_audio import convert_text_to_audio
from modules.general_chatting.chat import return_chat
from modules.stock_market_sentiment.stock_sentiment import analyze_stock_sentiment
from modules.stock_market_sentiment.name_extractor import extract_company_name
from modules.gmail.gmail_main import gmail_operation
from intent_classifier.main import classify_intent
from modules.gmail.sub_intent_classifier.gmail_sub_intent_classifier import predict_sub_intent

# ------------------------ Weather Config --------------------------------
API_KEY = "671ec5dca70e76a538e9b9d3fc36182d"
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"
nlp = spacy.load("en_core_web_sm")

# ------------------------ Logging ---------------------------------------
logging.basicConfig(level=logging.INFO)

# ------------------------ Streamlit Config ------------------------------
st.set_page_config(page_title="🤖 Multi-Purpose AI Agent", layout="wide")

# ======================== UI STYLE ====================
st.markdown("""
<style>
/* Dark overlay for background images */
.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.45);
    z-index: -1;
}

/* Scrollable chat container */
.chat-container {
    max-height: 70vh;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 12px;
    width: 100%;
    padding: 10px;
}

/* User prompt - left aligned */
.user-prompt-full {
    background-color: rgba(30,30,30,0.95);  /* original color */
    color: white;
    padding: 14px;
    width: 100%;
    font-weight: bold;
    border-radius: 12px;
    text-align: right;   /* right alignment */
    word-break: break-word;
    font-size: 16px;
    margin-right: 0;    /* align bubble to left */
    max-width: 100%;     /* bubble width limit */
}

/* AI response - left aligned */
.ai-msg-full {
    background: rgba(20,20,20,0.85);
    color: #f1f1f1;
    padding: 14px;
    border-radius: 12px;
    font-weight: bold;
    text-align: left;
    word-break: break-word;
    font-size: 16px;
}

/* Intent badge */
.intent-badge-full {
    background-color: #ffcc00;
    color: black;
    padding: 6px 12px;
    border-radius: 12px;
    font-weight: bold;
    display: inline-block;
    margin-bottom: 6px;
}

/* Input pinned at bottom */
#input-container {
    position: fixed;
    bottom: 0;
    width: 100%;
    background-color: rgba(30,30,30,0.95);
    padding: 10px 20px;
    display: flex;
    gap: 10px;
    align-items: center;
    z-index: 999;
}

/* Input fields */
input, textarea, button {
    border-radius: 10px !important;
    padding: 8px;
    font-size: 16px;
}
input, textarea {
    flex: 1;
    background-color: rgba(0,0,0,0.75) !important;
    color: white !important;
    border: 1px solid #555;
}
button {
    background-color: #007bff;
    color: white;
    border: none;
    cursor: pointer;
}
button:hover {
    background-color: #0056b3;
}
</style>
""", unsafe_allow_html=True)

# ==================== BACKGROUND IMAGE ==============================
def set_bg_from_local(img_path):
    with open(img_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpeg;base64,{encoded}");
                background-size: cover;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

bg_folder = "backgrounds"
bg_files = []
if os.path.exists(bg_folder):
    bg_files = [f for f in os.listdir(bg_folder) if f.endswith((".jpg", ".jpeg", ".png"))]

bg_files.insert(0, "None")
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    selected_bg = st.selectbox("Choose Background", bg_files)
    classifier_type = st.selectbox(
        "Select Intent Classifier",
        ["ml", "rule_based", "transformer"],
        index=0
    )
    if selected_bg != "None":
        set_bg_from_local(os.path.join(bg_folder, selected_bg))

# ========================= Weather Functions ==========================
def extract_city(prompt: str) -> str:
    doc = nlp(prompt.title())
    for ent in doc.ents:
        if ent.label_ == "GPE":
            return ent.text
    return "Pune"

def get_weather(prompt: str) -> str:
    city = extract_city(prompt)
    params = {"q": city, "appid": API_KEY, "units": "metric"}
    try:
        response = requests.get(BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        weather_str = f"""🌤️ Weather in **{city}** \n
        • Condition: {data['weather'][0]['description'].capitalize()} \n
        • Temperature: {data['main']['temp']} °C \n
        • Feels Like: {data['main']['feels_like']} °C \n
        • Humidity: {data['main']['humidity']} % \n
        • Wind Speed: {data['wind']['speed']} m/s \n"""
        return weather_str
    except Exception as e:
        return f"❌ Weather fetch failed: {e}"

# ========================= Stock Sentiment Pie ========================
def plot_sentiment_pie(df):
    sentiment_col = next((c for c in df.columns if c.lower() == "sentiment"), None)
    if sentiment_col is None:
        return
    counts = df[sentiment_col].value_counts()
    fig, ax = plt.subplots()
    ax.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90)
    ax.axis("equal")
    st.pyplot(fig)

# ========================= UI HEADER ================================
st.title("🤖 Multi-Purpose AI Agent")
st.caption("Chat • Weather • Stock Sentiment • Smart Summaries • Text-to-Speech • Gmail • NL2SQL")

# ========================= MESSAGE CONTAINER =========================
chat_container = st.container()

# ========================= INPUT FORM (BOTTOM) =======================
with st.container():
    with st.form("chat_form", clear_on_submit=True):
        prompt = st.text_input("Enter your message", key="input_text")
        uploaded_file = st.file_uploader("Optional Attachment", type=["jpg","jpeg","png","mp3","wav","m4a"])
        submit = st.form_submit_button("Send")

# ========================= MAIN LOGIC ================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if submit and prompt:
    try:
        intent = classify_intent(prompt, method=classifier_type)
    except Exception:
        intent = "Unknown"

    # Append user message
    st.session_state.chat_history.append({
        "role": "user",
        "message": prompt,
        "intent": intent
    })

    # Generate AI response
    ai_message = ""
    audio_file = None
    notes_file = None
    df_stock = None

    if intent == "make_notes" and uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded_file.read())
            notes = make_notes_from_image(tmp.name)
            os.remove(tmp.name)
        ai_message = notes
        notes_file = "notes.mp3"
        convert_text_to_audio(notes, notes_file)

    elif intent == "convert_to_audio":
        ai_message = "🔊 Converted your text to audio!"
        audio_file = "speech.mp3"
        convert_text_to_audio(prompt, audio_file)

    elif intent == "get_weather":
        ai_message = get_weather(prompt)

    elif intent == "stock_sentiment":
        company, url = extract_company_name(prompt)
        if company and url:
            df_stock = analyze_stock_sentiment(url)
            ai_message = f"📊 Stock sentiment for {company}"
        else:
            ai_message = "Company not detected"

    elif intent == "gmail_operations":
        results = gmail_operation(prompt)
        ai_message = ""
        for mail in results:
            ai_message += f"**{mail['Subject']}**\n{mail['Body']}\n\n"

    else:
        ai_message = return_chat(prompt)

    # Append AI response
    st.session_state.chat_history.append({
        "role": "ai",
        "message": ai_message,
        "intent": intent,
        "audio_file": audio_file,
        "df_stock": df_stock
    })

# ========================= DISPLAY CHAT HISTORY ======================
with chat_container:
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for chat in st.session_state.chat_history:
        # User
        if chat["role"] == "user":
            st.markdown(f"<div class='user-prompt-full'><b>Your prompt :</b> {chat['message']}</div>", unsafe_allow_html=True)
            st.markdown(f"<span class='intent-badge-full'>Intent: {chat['intent']}</span>", unsafe_allow_html=True)
        # AI
        if chat["role"] == "ai":
            st.markdown(f"<div class='ai-msg-full'>{chat['message']}</div>", unsafe_allow_html=True)
            if chat.get("audio_file"):
                st.audio(chat["audio_file"])
            if chat.get("df_stock") is not None:
                st.dataframe(chat["df_stock"])
                plot_sentiment_pie(chat["df_stock"])
    st.markdown("</div>", unsafe_allow_html=True)





################################################################################################

# # ------------------------ Import necessary libraries -------------------
# import streamlit as st
# import logging
# import tempfile
# import os
# import base64
# import requests
# import spacy
# import matplotlib.pyplot as plt
# import pandas as pd

# # ------------------------ Internal Modules -----------------------------
# from modules.notes_maker.notes_maker import make_notes_from_image
# from modules.text_to_audio.text_to_audio import convert_text_to_audio
# from modules.general_chatting.chat import return_chat
# from modules.stock_market_sentiment.stock_sentiment import analyze_stock_sentiment
# from modules.stock_market_sentiment.name_extractor import extract_company_name
# from modules.gmail.gmail_main import gmail_operation
# from intent_classifier.main import classify_intent
# from modules.gmail.sub_intent_classifier.gmail_sub_intent_classifier import predict_sub_intent

# # ------------------------ Weather Config --------------------------------
# API_KEY = "671ec5dca70e76a538e9b9d3fc36182d"
# BASE_URL = "https://api.openweathermap.org/data/2.5/weather"
# nlp = spacy.load("en_core_web_sm")

# # ------------------------ Logging ---------------------------------------
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# # ------------------------ Streamlit Config ------------------------------
# st.set_page_config(page_title="AI Agent", layout="centered")

# # ------------------------ Sidebar ---------------------------------------


# def set_bg_from_local(img_path):
#     with open(img_path, "rb") as image_file:
#         encoded = base64.b64encode(image_file.read()).decode()
#         st.markdown(
#             f"""
#             <style>
#             .stApp {{
#                 background-image: url("data:image/jpeg;base64,{encoded}");
#                 background-size: cover;
#             }}
#             </style>
#             """,
#             unsafe_allow_html=True
#         )

# bg_folder = "backgrounds"
# bg_files = [f for f in os.listdir(bg_folder) if f.endswith((".jpg", ".jpeg", ".png"))]
# bg_files.insert(0, "None")
# selected_bg = st.sidebar.selectbox("Choose Background", bg_files)

# if selected_bg != "None":
#     set_bg_from_local(os.path.join(bg_folder, selected_bg))

# with st.sidebar:
#     st.markdown("### ⚙️ Settings")
#     classifier_type = st.selectbox(
#         "Select Intent Classifier",
#         ["ml", "rule_based", "transformer"],
#         index=0
#     )

# # ------------------------ Weather Functions -----------------------------
# def extract_city(prompt: str) -> str:
#     doc = nlp(prompt.title())
#     for ent in doc.ents:
#         if ent.label_ == "GPE":
#             return ent.text
#     return "Pune"

# def get_weather(prompt: str) -> str:
#     city = extract_city(prompt)
#     params = {"q": city, "appid": API_KEY, "units": "metric"}

#     try:
#         response = requests.get(BASE_URL, params=params, timeout=10)
#         response.raise_for_status()
#         data = response.json()

#         return (
#             f"🌤️ **Weather Update for {city}**\n\n"
#             f"- **Condition:** {data['weather'][0]['description'].capitalize()}\n"
#             f"- **Temperature:** {data['main']['temp']} °C\n"
#             f"- **Feels Like:** {data['main']['feels_like']} °C\n"
#             f"- **Humidity:** {data['main']['humidity']}%\n"
#             f"- **Wind Speed:** {data['wind']['speed']} m/s"
#         )
#     except Exception as e:
#         return f"❌ Weather fetch failed: {e}"

# # ------------------------ Stock Sentiment Pie Chart ---------------------
# def plot_sentiment_pie(df: pd.DataFrame):
#     sentiment_col = None
#     for col in df.columns:
#         if col.lower() == "sentiment":
#             sentiment_col = col
#             break

#     if sentiment_col is None:
#         st.warning("Sentiment column not found.")
#         return

#     counts = df[sentiment_col].value_counts()

#     fig, ax = plt.subplots()
#     ax.pie(
#         counts.values,
#         labels=counts.index,
#         autopct="%1.1f%%",
#         startangle=90
#     )
#     ax.axis("equal")
#     ax.set_title("Sentiment Distribution")

#     st.pyplot(fig)

# # ------------------------ UI Header -------------------------------------
# # st.title("Chat with Multi-Purpose AI Agent")
# st.title("Let's Chat")

# # ------------------------ Chat Input ------------------------------------
# with st.form("chat_form"):
#     prompt = st.text_input("Enter your message:")
#     uploaded_file = st.file_uploader(
#         "Optional Attachment (Image/Audio)",
#         type=["jpg", "jpeg", "png", "mp3", "wav", "m4a"]
#     )
#     submit = st.form_submit_button("Send")

# # ------------------------ Intent Router ---------------------------------
# if submit and prompt:
#     st.markdown(f"### You: {prompt}")

#     try:
#         intent = classify_intent(prompt, method=classifier_type)
#         st.markdown(f"_Intent Detected: `{intent}`_")
#     except Exception:
#         intent = None
#         st.error("Intent classification failed.")

#     # ---------------- Notes Maker ----------------
#     if intent == "make_notes":
#         if uploaded_file and uploaded_file.type.startswith("image/"):
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
#                 tmp.write(uploaded_file.read())
#                 path = tmp.name
#             notes = make_notes_from_image(path)
#             st.text_area("Notes", notes, height=200)
#             convert_text_to_audio(notes, "notes.mp3")
#             st.audio("notes.mp3")
#             os.remove(path)

#     # ---------------- Text to Audio --------------
#     elif intent == "convert_to_audio":
#         convert_text_to_audio(prompt, "speech.mp3")
#         st.audio("speech.mp3")

#     # ---------------- Weather --------------------
#     elif intent == "get_weather":
#         st.subheader("🌦️ Weather Information")
#         st.markdown(get_weather(prompt))

#     # ---------------- Stock Sentiment (PIE CHART) ----------------
#     elif intent == "stock_sentiment":
#         st.subheader("📈 Stock Market Sentiment Analysis")

#         try:
#             company, news_url = extract_company_name(prompt)

#             if not company or not news_url:
#                 st.warning("Could not extract company name.")
#             else:
#                 df = analyze_stock_sentiment(news_url)

#                 if df.empty:
#                     st.warning("No sentiment data found.")
#                 else:
#                     st.success(f"Sentiment for `{company}`")
#                     st.dataframe(df)

#                     st.markdown("### 📊 Sentiment Distribution")
#                     plot_sentiment_pie(df)

#         except Exception:
#             st.error("Stock sentiment analysis failed.")
#             logging.error("Stock sentiment error", exc_info=True)

#     # ---------------- Gmail ----------------------
#     elif intent == "gmail_operations":
#         sub_intent = predict_sub_intent(prompt)
#         st.markdown(f"_Sub-Intent: `{sub_intent}`_")

#         results = gmail_operation(prompt)
#         for i, email in enumerate(results, 1):
#             st.markdown(f"### Email {i}")
#             st.markdown(f"**From:** {email['From']}")
#             st.markdown(f"**Subject:** {email['Subject']}")
#             st.text_area("Body", email["Body"], height=150, key=f"mail_{i}")

#     # ---------------- General Chat ---------------
#     elif intent == "general_chat":
#         st.success(return_chat(prompt))

#     # ---------------- Fallback -------------------
#     else:
#         st.warning("Unsupported or unknown intent.")






# # # ------------------------ Import necessary libraries -------------------
# # import streamlit as st
# # import logging
# # import tempfile
# # import os
# # import base64

# # # ------------------------ Internal Modules -----------------------------
# # from modules.notes_maker.notes_maker import make_notes_from_image
# # from modules.text_to_audio.text_to_audio import convert_text_to_audio
# # from modules.general_chatting.chat import return_chat
# # from modules.stock_market_sentiment.stock_sentiment import analyze_stock_sentiment
# # from modules.stock_market_sentiment.name_extractor import extract_company_name
# # from modules.gmail.gmail_main import gmail_operation
# # from intent_classifier.main import classify_intent
# # from modules.gmail.sub_intent_classifier.gmail_sub_intent_classifier import predict_sub_intent

# # # ------------------------ Weather Module --------------------------------
# # import requests
# # import spacy

# # API_KEY = "671ec5dca70e76a538e9b9d3fc36182d"
# # BASE_URL = "https://api.openweathermap.org/data/2.5/weather"
# # nlp = spacy.load("en_core_web_sm")

# # # ------------------------ Logging ---------------------------------------
# # logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# # # ------------------------ Streamlit Config ------------------------------
# # st.set_page_config(page_title="AI Agent", layout="centered")

# # # ------------------------ Sidebar Aesthetics ----------------------------


# # def set_bg_from_local(img_path):
# #     with open(img_path, "rb") as image_file:
# #         encoded_string = base64.b64encode(image_file.read()).decode()
# #         css = f"""
# #         <style>
# #         .stApp {{
# #             background-image: url("data:image/jpeg;base64,{encoded_string}");
# #             background-size: cover;
# #         }}
# #         </style>
# #         """
# #         st.markdown(css, unsafe_allow_html=True)

# # bg_folder = "backgrounds"
# # bg_files = [f for f in os.listdir(bg_folder) if f.endswith((".jpg", ".jpeg", ".png"))]

# # default_image = "Default.jpg"
# # if default_image in bg_files:
# #     bg_files.remove(default_image)
# #     bg_files.insert(0, default_image)

# # bg_files.insert(0, "None")
# # selected_bg = st.sidebar.selectbox("Choose Background", bg_files, index=1)

# # if selected_bg != "None":
# #     set_bg_from_local(os.path.join(bg_folder, selected_bg))

# # with st.sidebar:
# #     st.markdown("### ⚙️ Settings")
# #     classifier_type = st.selectbox(
# #         "Select Intent Classifier",
# #         ["ml", "rule_based", "transformer"],
# #         index=0
# #     )

# # # ------------------------ Weather Functions -----------------------------
# # def extract_city(prompt: str) -> str:
# #     doc = nlp(prompt.title())
# #     for ent in doc.ents:
# #         if ent.label_ == "GPE":
# #             return ent.text
# #     return "Pune"

# # def get_weather(prompt: str) -> str:
# #     city_name = extract_city(prompt)

# #     params = {
# #         "q": city_name,
# #         "appid": API_KEY,
# #         "units": "metric"
# #     }

# #     try:
# #         response = requests.get(BASE_URL, params=params, timeout=10)
# #         response.raise_for_status()
# #         data = response.json()

# #         return (
# #             f"🌤️ **Weather Update for {city_name}**\n\n"
# #             f"- **Condition:** {data['weather'][0]['description'].capitalize()}\n"
# #             f"- **Temperature:** {data['main']['temp']} °C\n"
# #             f"- **Feels Like:** {data['main']['feels_like']} °C\n"
# #             f"- **Humidity:** {data['main']['humidity']}%\n"
# #             f"- **Wind Speed:** {data['wind']['speed']} m/s"
# #         )

# #     except requests.exceptions.RequestException as e:
# #         return f"❌ Failed to get weather data: {e}"

# # # ------------------------ UI Header -------------------------------------
# # st.title("Chat with Multi-Purpose AI Agent")

# # # ------------------------ Chat Form -------------------------------------
# # with st.form("chat_form"):
# #     prompt = st.text_input("Enter your message:")
# #     uploaded_file = st.file_uploader(
# #         "Optional Attachment (Image/Audio)",
# #         type=["jpg", "jpeg", "png", "mp3", "wav", "m4a"]
# #     )
# #     submit = st.form_submit_button("Send")

# # # ------------------------ Intent Routing --------------------------------
# # if submit and prompt:
# #     st.markdown(f"### You: {prompt}")

# #     try:
# #         intent = classify_intent(prompt, method=classifier_type)
# #         st.markdown(f"_Intent Detected: `{intent}`_")
# #     except Exception:
# #         intent = None
# #         st.error("Intent classification failed.")
# #         logging.error("Intent classification error", exc_info=True)

# #     # -------------------- Notes from Image -------------------------------
# #     if intent == "make_notes":
# #         st.subheader("📝 Note Maker from Image")

# #         if uploaded_file and uploaded_file.type.startswith("image/"):
# #             with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
# #                 tmp.write(uploaded_file.read())
# #                 image_path = tmp.name

# #             try:
# #                 notes = make_notes_from_image(image_path)
# #                 st.text_area("Extracted Notes", notes, height=200)

# #                 audio_path = "notes_audio.mp3"
# #                 convert_text_to_audio(notes, audio_path)
# #                 st.audio(audio_path)

# #             except Exception:
# #                 st.error("Failed to process image.")
# #                 logging.error("Notes module failed", exc_info=True)

# #             finally:
# #                 os.remove(image_path)
# #         else:
# #             st.warning("Upload a valid image file.")

# #     # -------------------- Text to Audio ----------------------------------
# #     elif intent == "convert_to_audio":
# #         st.subheader("🔊 Text to Speech")

# #         try:
# #             convert_text_to_audio(prompt, "converted_audio.mp3")
# #             st.audio("converted_audio.mp3")
# #         except Exception:
# #             st.error("Text-to-speech failed.")
# #             logging.error("TTS error", exc_info=True)

# #     # -------------------- Weather ---------------------------------------
# #     elif intent == "get_weather":
# #         st.subheader("🌦️ Weather Information")
# #         st.markdown(get_weather(prompt))

# #     # -------------------- Stock Sentiment --------------------------------
# #     elif intent == "stock_sentiment":
# #         st.subheader("📈 Stock Market Sentiment")

# #         try:
# #             company_name, news_url = extract_company_name(prompt)
# #             if not company_name:
# #                 st.warning("Company name not detected.")
# #             else:
# #                 df = analyze_stock_sentiment(news_url)
# #                 st.dataframe(df)
# #         except Exception:
# #             st.error("Stock sentiment analysis failed.")
# #             logging.error("Stock sentiment error", exc_info=True)

# #     # -------------------- Gmail Operations -------------------------------
# #     elif intent == "gmail_operations":
# #         st.subheader("📧 Gmail Operations")

# #         try:
# #             sub_intent = predict_sub_intent(prompt)
# #             st.markdown(f"_Sub-Intent: `{sub_intent}`_")

# #             results = gmail_operation(prompt)

# #             for i, email in enumerate(results, 1):
# #                 st.markdown(f"### Email {i}")
# #                 st.markdown(f"**From:** {email['From']}")
# #                 st.markdown(f"**Subject:** {email['Subject']}")
# #                 st.markdown(f"**Date:** {email['Date']}")
# #                 st.text_area("Body", email["Body"], height=150, key=f"email_{i}")

# #         except Exception:
# #             st.error("Gmail operation failed.")
# #             logging.error("Gmail error", exc_info=True)

# #     # -------------------- General Chat -----------------------------------
# #     elif intent == "general_chat":
# #         st.subheader("💬 General Chat")

# #         try:
# #             response = return_chat(prompt)
# #             st.success(response)
# #         except Exception:
# #             st.error("Chat response failed.")
# #             logging.error("Chat error", exc_info=True)

# #     # -------------------- Fallback --------------------------------------
# #     else:
# #         st.warning("Unsupported or unknown intent.")


# # # # ------------------------Import necessary libraries-------------------
# # # import streamlit as st
# # # import logging
# # # import tempfile
# # # import os
# # # from modules.notes_maker.notes_maker import make_notes_from_image
# # # from modules.text_to_audio.text_to_audio import convert_text_to_audio
# # # from modules.general_chatting.chat import return_chat
# # # from modules.stock_market_sentiment.stock_sentiment import analyze_stock_sentiment
# # # from modules.stock_market_sentiment.name_extractor import extract_company_name
# # # from modules.gmail.gmail_main import gmail_operation
# # # import base64
# # # from intent_classifier.main import classify_intent
# # # from modules.gmail.sub_intent_classifier.gmail_sub_intent_classifier import predict_sub_intent

# # # logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# # # st.set_page_config(page_title="AI Agent", layout="centered")

# # # # ---------------------------------asthetics-------------------------------------------
# # # st.sidebar.markdown("**[🔗 LinkedIn- Jay Vijay Sawant](https://www.linkedin.com/in/jay-sawant-0011/)**", unsafe_allow_html=True)
# # # def set_bg_from_local(img_path):
# # #     with open(img_path, "rb") as image_file:
# # #         encoded_string = base64.b64encode(image_file.read()).decode()
# # #         css = f"""
# # #         <style>
# # #         .stApp {{
# # #             background-image: url("data:image/jpeg;base64,{encoded_string}");
# # #             background-size: cover;
# # #         }}
# # #         </style>
# # #         """
# # #         st.markdown(css, unsafe_allow_html=True)

# # # bg_folder = "backgrounds"
# # # bg_files = [f for f in os.listdir(bg_folder) if f.endswith((".jpg", ".jpeg", ".png"))]

# # # default_image = "Default.jpg"

# # # if default_image in bg_files:
# # #     bg_files.remove(default_image)
# # #     bg_files.insert(0, default_image)
# # # bg_files.insert(0, "None")

# # # selected_bg = st.sidebar.selectbox("Choose Background", bg_files, index=1)

# # # if selected_bg != "None":
# # #     image_path = os.path.join(bg_folder, selected_bg)
# # #     set_bg_from_local(image_path)

# # # with st.sidebar:
# # #     st.markdown("### ⚙️ Settings")
# # #     classifier_type = st.selectbox(
# # #         "Select Intent Classifier",
# # #         ["ml", "rule_based", "transformer"],
# # #         index=0,
# # #         help="Choose the method for intent classification"
# # #     )
    

    


# # # st.title("Chat with Multi-Purpose AI Agent")
# # # st.caption("Legends don’t use templates — Built from scratch by Jay & Jeswin 🔥")


# # # # -----------------------chat------------------

# # # with st.form("chat_form"):
# # #     prompt = st.text_input("Enter your message:")
# # #     uploaded_file = st.file_uploader("Optional Attachment (Image/Audio)", type=["jpg", "jpeg", "png", "mp3", "wav", "m4a"])
# # #     submit = st.form_submit_button("Send")

# # # if submit and prompt:
# # #     st.markdown(f"### You: {prompt}")

# # #     try:
# # #         intent = classify_intent(prompt, method=classifier_type)

# # #         st.markdown(f"_Intent Detected: `{intent}`_")
# # #     except Exception as e:
# # #         st.error("Intent classification failed.")
# # #         logging.error("Error during intent classification", exc_info=True)
# # #         intent = None

# # #     if intent == "make_notes":
# # #         st.subheader("Note Maker from Image")
# # #         if uploaded_file and uploaded_file.type.startswith("image/"):
# # #             try:
# # #                 with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img:
# # #                     tmp_img.write(uploaded_file.read())
# # #                     image_path = tmp_img.name

# # #                 notes = make_notes_from_image(image_path)
# # #                 st.text_area("Extracted Notes", value=notes, height=200)

# # #                 audio_path = "notes_audio.mp3"
# # #                 convert_text_to_audio(notes, audio_path)
# # #                 st.audio(audio_path)
# # #             except Exception as e:
# # #                 st.error("Failed to extract notes or generate audio.")
# # #                 logging.error("Error in make_notes workflow", exc_info=True)
# # #             finally:
# # #                 os.remove(image_path)
# # #         else:
# # #             st.error("Please upload a valid image file.")

# # #     elif intent == "convert_to_audio":
# # #         st.subheader("Text to Speech")
# # #         try:
# # #             convert_text_to_audio(prompt, "converted_audio.mp3")
# # #             st.audio("converted_audio.mp3")
# # #         except Exception as e:
# # #             st.error("Failed to convert text to audio.")
# # #             logging.error("Text to audio conversion failed", exc_info=True)

# # #     elif intent == "stock_sentiment":
# # #         st.subheader("Stock Market Sentiment Analysis")
# # #         try:
# # #             company_name, news_url = extract_company_name(prompt)
# # #             if not company_name or not news_url:
# # #                 st.warning("Could not extract a valid company name. Try again.")
# # #             else:
# # #                 df = analyze_stock_sentiment(news_url)
# # #                 if df.empty:
# # #                     st.warning(f"No news articles found for `{company_name}`.")
# # #                 else:
# # #                     st.success(f"News Sentiment for `{company_name}`")
# # #                     st.dataframe(df)
# # #         except Exception as e:
# # #             st.error("Error while fetching sentiment.")
# # #             logging.error("Stock sentiment analysis failed", exc_info=True)

# # #     elif intent == "general_chat":
# # #         st.subheader("General Chat")
# # #         try:
# # #             response = return_chat(prompt)
# # #             st.success("Response:")
# # #             st.write(response)
# # #         except Exception as e:
# # #             st.error("Failed to generate chat response.")
# # #             logging.error("General chat error", exc_info=True)

# # #     elif intent == "voice_summary":
# # #         st.subheader("Audio Summarization (Coming Soon)")
# # #         st.info("Audio summarization is not yet implemented.")

# # #     elif intent == "gmail_operations":
# # #         st.subheader("Gmail Operations")
# # #         try:
# # #             sub_intent = predict_sub_intent(prompt)
# # #             st.markdown(f"_Gmail Sub-Intent: `{sub_intent}`_")
# # #             result = gmail_operation(prompt)
# # #             for idx, email in enumerate(result, 1):
# # #                 st.markdown(f"#### Email {idx}")
# # #                 st.markdown(f"- **From:** {email['From']}")
# # #                 st.markdown(f"- **Subject:** {email['Subject']}")
# # #                 st.markdown(f"- **Date:** {email['Date']}")
# # #                 st.markdown(f"- **Snippet:** {email['Snippet']}")
# # #                 st.text_area(f"Body of Email {idx}", email['Body'], height=150, key=f"email_body_{idx}")

# # #             else:
# # #                 st.success(result)
# # #         except Exception as e:
# # #             st.error("Gmail operation failed.")
# # #             logging.error("Gmail module failed", exc_info=True)

# # #     else:
# # #         st.warning("Unknown or unsupported intent.")

