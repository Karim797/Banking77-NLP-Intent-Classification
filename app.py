import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


TRAIN_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv"

st.set_page_config(page_title="Banking77 Intent Classifier", page_icon="🏦")


@st.cache_resource
def train_model():
    data = pd.read_csv(TRAIN_URL)
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, strip_accents="unicode")),
        ("model", LogisticRegression(C=10, max_iter=2000)),
    ])
    model.fit(data["text"], data["category"])
    return model


st.title("Banking77 Intent Classification")
st.caption("Classify a customer message into one of 77 banking intents.")
model = train_model()
message = st.text_area("Customer message", "My transfer has not arrived yet")

if st.button("Classify intent") and message.strip():
    probabilities = model.predict_proba([message])[0]
    classes = model.classes_
    top = probabilities.argsort()[::-1][:3]
    st.success(f"Predicted intent: {classes[top[0]]}")
    st.metric("Confidence", f"{probabilities[top[0]]:.1%}")
    st.write("Top alternatives")
    st.dataframe(pd.DataFrame({"intent": classes[top], "probability": probabilities[top]}), hide_index=True)

