import streamlit as st
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from transformers_interpret import SequenceClassificationExplainer

@st.cache(allow_output_mutation=True)
def get_model():
    model_name = "samanjoy2/banglaclickbert_finetuned_sequence_classification_clickbait"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = get_model()

st.title('Bangla Clickbait Detection :sunglasses:')
st.header('Check whether a Bangla News Headline is Clickbait or not')

# Define the sidebar buttons and their corresponding text
clickbait_examples = [
    "অবশেষে মুখ খুললেন রাজ, যা বললেন পরীমনি ও সন্তান প্রসঙ্গে",
    "পান্তা ভাতের ইংরেজি কী জানেন? খুব সহজ! মাথায় আসছে না তো! জানুন",
    "এক টানেই আধ কোটির ইলিশ! সাগরে জাল ফেলতেই ভাগ্য বদল মৎস্যজীবীর! ইলিশে ইলিশে ছেয়ে গেল বাজার!"
]

non_clickbait_examples = [
    "যুক্তরাজ্যে ফ্লাইট বিপর্যয়, ভোগান্তি থাকবে ‘কয়েকদিন’",
    "চট্টগ্রামে খাল–নালায় মানুষ মরছে, তবু নিরাপত্তাবেষ্টনী উঠছে না",
    "বায়ুদূষণে বাংলাদেশের মানুষের গড় আয়ু কমছে প্রায় ৭ বছর"
]

# Create buttons in the sidebar for clickbait and non-clickbait examples
with st.sidebar:
    st.subheader('Clickbait Examples:')
    for example in clickbait_examples:
        if st.button(example):
            st.session_state.user_input = example
    st.write('---')
    st.subheader('Non-Clickbait Examples:')
    for example in non_clickbait_examples:
        if st.button(example):
            st.session_state.user_input = example

user_input = st.text_area('Enter Text to Analyze', key="user_input")
button = st.button("Analyze")
st.write('---')

if user_input and button:
    cls_explainer = SequenceClassificationExplainer(model, tokenizer)
    word_attributions = cls_explainer(user_input)
    st.write(cls_explainer.visualize())
    # st.divider()
    if cls_explainer.predicted_class_index == 1:
        st.subheader('Label Predicted: _Clickbait_')
    else:
        st.subheader('Label Predicted: _Not-Clickbait_')


url = "https://share.streamlit.io/mesmith027/streamlit_webapps/main/MC_pi/streamlit_app.py"
st.write("check out this [link](%s)" % url)

st.markdown("check out this [link](%s)" % url)
