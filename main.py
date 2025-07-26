import os
import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Fallback in case sentencepiece is missing
try:
    import sentencepiece
except ImportError:
    os.system("pip install sentencepiece==0.1.99")

from transformers import pipeline

# --- Globals ---
ai_model, lang_model = None, None
ai_vectorizer, lang_vectorizer = None, None

# --- Translation Config ---
LANGUAGE_CODES = {
    "German": "de", "French": "fr", "Spanish": "es", "Hindi": "hi", "Arabic": "ar",
    "Chinese": "zh", "Japanese": "ja", "Russian": "ru", "Italian": "it", "Portuguese": "pt"
}

@st.cache_resource
def load_translator(model_name):
    return pipeline("translation", model=model_name)

def translate_text(text, target_lang):
    if target_lang not in LANGUAGE_CODES:
        return "Unsupported language."
    tgt_code = LANGUAGE_CODES[target_lang]
    model_name = f"Helsinki-NLP/opus-mt-en-{tgt_code}"
    try:
        translator = load_translator(model_name)
        result = translator(text, max_length=512)
        return result[0]["translation_text"]
    except Exception as e:
        return f"Translation Error: {e}"

# --- AI Model Training ---
def train_ai_model(df):
    global ai_model, ai_vectorizer
    if 'Text' not in df.columns or 'Label' not in df.columns:
        return "❌ CSV must contain 'Text' and 'Label'."
    X, y = df['Text'], df['Label']
    ai_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), stop_words='english')
    X_vec = ai_vectorizer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
    ai_model = LogisticRegression()
    ai_model.fit(X_train, y_train)
    acc = ai_model.score(X_test, y_test)
    pickle.dump(ai_model, open("clf.pkl", "wb"))
    pickle.dump(ai_vectorizer, open("tfidf.pkl", "wb"))
    return f"✅ AI model trained with {round(acc*100, 2)}% accuracy."

# --- Language Model Training ---
def train_lang_model(df):
    global lang_model, lang_vectorizer
    if 'Text' not in df.columns or 'Language' not in df.columns:
        return "❌ CSV must contain 'Text' and 'Language'."
    X, y = df['Text'], df['Language']
    lang_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
    X_vec = lang_vectorizer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
    lang_model = LogisticRegression(max_iter=1000)
    lang_model.fit(X_train, y_train)
    acc = lang_model.score(X_test, y_test)
    pickle.dump(lang_model, open("language_model.pkl", "wb"))
    pickle.dump(lang_vectorizer, open("tfidf_vectorizer.pkl", "wb"))
    return f"✅ Language model trained with {round(acc*100, 2)}% accuracy."

# --- Load Trained Models ---
def load_models():
    global ai_model, ai_vectorizer, lang_model, lang_vectorizer
    if os.path.exists("clf.pkl") and os.path.exists("tfidf.pkl"):
        ai_model = pickle.load(open("clf.pkl", "rb"))
        ai_vectorizer = pickle.load(open("tfidf.pkl", "rb"))
    if os.path.exists("language_model.pkl") and os.path.exists("tfidf_vectorizer.pkl"):
        lang_model = pickle.load(open("language_model.pkl", "rb"))
        lang_vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# --- AI Prediction ---
def predict_ai(text):
    if ai_model is None or ai_vectorizer is None:
        return "⚠️ AI model not trained yet."
    vec = ai_vectorizer.transform([text])
    pred = ai_model.predict(vec)[0]
    return "🤖 AI Generated" if pred == 1 else "👤 Human Written"

# --- Language Prediction ---
def predict_lang(text):
    if lang_model is None or lang_vectorizer is None:
        return "⚠️ Language model not trained yet."
    vec = lang_vectorizer.transform([text])
    pred = lang_model.predict(vec)[0]
    return f"🌍 Detected Language: {pred}"

# --- Streamlit UI ---
st.set_page_config(page_title="Multilingual AI Text Checker", layout="centered")
st.title("🧠 Multilingual AI Text Checker")

load_models()

tab1, tab2, tab3 = st.tabs(["📚 Train Models", "🔍 Predict Text", "🌐 Translate"])

# === Tab 1: Training ===
with tab1:
    st.subheader("Train AI & Language Detection Models")
    
    ai_file = st.file_uploader("Upload AI CSV ('Text', 'Label')", type='csv')
    if st.button("Train AI Detector"):
        if ai_file:
            df = pd.read_csv(ai_file)
            st.success(train_ai_model(df))
        else:
            st.warning("Upload a valid AI CSV")

    lang_file = st.file_uploader("Upload Language CSV ('Text', 'Language')", type='csv')
    if st.button("Train Language Detector"):
        if lang_file:
            df = pd.read_csv(lang_file)
            st.success(train_lang_model(df))
        else:
            st.warning("Upload a valid Language CSV")

# === Tab 2: Prediction ===
with tab2:
    st.subheader("Enter Text for AI/Human and Language Detection")
    user_input = st.text_area("Enter text here:")
    col1, col2 = st.columns(2)

    if col1.button("🧠 AI vs Human"):
        st.info(predict_ai(user_input))

    if col2.button("🌍 Detect Language"):
        st.success(predict_lang(user_input))

    if st.button("✨ Predict Both"):
        st.info(predict_ai(user_input))
        st.success(predict_lang(user_input))

# === Tab 3: Translation ===
with tab3:
    st.subheader("Translate English Text into Other Languages")
    eng_text = st.text_area("Enter English Text to Translate:")
    lang_choice = st.selectbox("Translate To", list(LANGUAGE_CODES.keys()))

    if st.button("🔁 Translate"):
        st.success(translate_text(eng_text, lang_choice))
