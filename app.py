import streamlit as st
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from transformers_interpret import SequenceClassificationExplainer

@st.cache_data()
def get_model():
    model_name = "samanjoy2/banglaclickbert_finetuned_sequence_classification_clickbait"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer,model = get_model()

st.title('Bangla Clickbait Detection :sunglasses:')
st.header('Check whether a Bangla News Headline is Clickbait or not', divider='rainbow')
user_input = st.text_area('Enter Text to Analyze')
button = st.button("Analyze")

if user_input and button :
    cls_explainer = SequenceClassificationExplainer(model, tokenizer)
    word_attributions = cls_explainer(user_input)
    st.write(cls_explainer.visualize())
    st.divider()
    if cls_explainer.predicted_class_index == 1:
        st.subheader('Label Predicted: _Clickbait_')
    else:
        st.subheader('Label Predicted: _Not-Clickbait_')
