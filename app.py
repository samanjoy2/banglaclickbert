import streamlit as st
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from transformers_interpret import SequenceClassificationExplainer

@st.cache_data()
def get_model():
    model_name = "samanjoy2/banglabert_finetuned_sequence_classification_clickbait"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer,model = get_model()

user_input = st.text_area('Enter Text to Analyze')
button = st.button("Analyze")

if user_input and button :
    cls_explainer = SequenceClassificationExplainer(model, tokenizer)
    word_attributions = cls_explainer(user_input)
    st.write(cls_explainer.visualize())
    if cls_explainer.predicted_class_index == 1:
        st.write('Lable: Clickbait')
    else:
        st.write('Lable: Not-Clickbait')