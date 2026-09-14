import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


MODEL_DIR = Path(__file__).with_name("artifacts") / "final_model"
MODEL_PATH = MODEL_DIR / "tfidf_logreg_pipeline.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"
LABELS_PATH = MODEL_DIR / "label_mapping.json"

st.set_page_config(page_title="Banking77 Intent Classifier", page_icon="🏦")


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    with METADATA_PATH.open(encoding="utf-8") as fh:
        metadata = json.load(fh)
    with LABELS_PATH.open(encoding="utf-8") as fh:
        labels = json.load(fh)
    id_to_name = {int(k): v for k, v in labels["id_to_name"].items()}
    return model, metadata, id_to_name


st.title("Banking77 Intent Classification")
st.caption("Classify a customer message into one of 77 banking intents.")
model, metadata, id_to_name = load_artifacts()
THRESHOLD = float(metadata["confidence_threshold"])
message = st.text_area("Customer message", "My transfer has not arrived yet")

if st.button("Classify intent") and message.strip():
    probabilities = model.predict_proba([message])[0]
    top = probabilities.argsort()[::-1][:3]
    class_ids = model.classes_[top]
    intents = [id_to_name[int(class_id)] for class_id in class_ids]
    confidence = float(probabilities[top[0]])

    if confidence < THRESHOLD:
        st.warning(
            f"Escalate to human review — confidence {confidence:.1%} is below "
            f"the validation-selected threshold of {THRESHOLD:.0%}."
        )
    else:
        st.success(f"Predicted intent: {intents[0]}")

    st.metric("Confidence", f"{confidence:.1%}")
    st.write("Top alternatives")
    st.dataframe(
        pd.DataFrame({"intent": intents, "probability": probabilities[top]}),
        hide_index=True,
    )
