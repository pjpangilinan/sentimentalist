import streamlit as st
import joblib
import numpy as np

vectorizer = joblib.load("vectorizer.pkl")

models = {
    "Logistic Regression": joblib.load("logistic_regression.pkl"),
    "Naive Bayes": joblib.load("naive_bayes.pkl"),
    "Random Forest": joblib.load("random_forest.pkl")
}

sentiment_map = {
    0: ("Negative", "#FF4B4B"),
    1: ("Neutral", "#888888"),
    2: ("Positive", "#4CAF50"),
}

st.set_page_config(page_title="Sentiment Classifier", layout="centered")

st.markdown("""
    <style>
    .main > div {
        background-color: #ffffffcc;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        max-width: 800px;
        margin: auto;
    }
    div[data-testid="stRadio"] > div {
        justify-content: center;
        gap: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <h2 style='text-align: center;'>🧠 Sentimentalist</h2>
""", unsafe_allow_html=True)

model_name = st.radio("🧪 Choose a model:", list(models.keys()), horizontal=True)

user_input = st.text_area(
    label="Comment Input",
    placeholder="💬 Enter your text here...",
    height=200,
    label_visibility="collapsed"
)

if user_input.strip():
    model = models[model_name]
    vectorized_input = vectorizer.transform([user_input])
    prediction = model.predict(vectorized_input)[0]
    probs = model.predict_proba(vectorized_input)[0]

    sentiment, color = sentiment_map.get(prediction, ("Unknown", "#000000"))
    confidence = np.max(probs) * 100

    st.markdown(f"""
        <div style='
            background-color: #ffffff;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            margin-top: 20px;
        '>
            <h3 style='color:{color}; margin-bottom: 0;'>{sentiment}</h3>
            <p style='color: #555;'>Confidence: {confidence:.2f}%</p>
        </div>
    """, unsafe_allow_html=True)
