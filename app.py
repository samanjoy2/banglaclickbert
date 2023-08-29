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

tokenizer,model = get_model()

with st.sidebar:
    st.subheader('Some Examples of Clickbait Headlines:')
    st.write("অবশেষে মুখ খুললেন রাজ, যা বললেন পরীমনি ও সন্তান প্রসঙ্গে")
    st.write("পান্তা ভাতের ইংরেজি কী জানেন? খুব সহজ! মাথায় আসছে না তো! জানুন")
    st.write("এক টানেই আধ কোটির ইলিশ! সাগরে জাল ফেলতেই ভাগ্য বদল মৎস্যজীবীর! ইলিশে ইলিশে ছেয়ে গেল বাজার!")
    st.subheader('Some Examples of Non-Clickbait Headlines:')
    st.write("যুক্তরাজ্যে ফ্লাইট বিপর্যয়, ভোগান্তি থাকবে ‘কয়েকদিন’")
    st.write("চট্টগ্রামে খাল–নালায় মানুষ মরছে, তবু নিরাপত্তাবেষ্টনী উঠছে না")
    st.write("বায়ুদূষণে বাংলাদেশের মানুষের গড় আয়ু কমছে প্রায় ৭ বছর")
    
    

st.title('Bangla Clickbait Detection :sunglasses:')
st.header('Check whether a Bangla News Headline is Clickbait or not')
user_input = st.text_area('Enter Text to Analyze')
button = st.button("Analyze")

if user_input and button :
    cls_explainer = SequenceClassificationExplainer(model, tokenizer)
    word_attributions = cls_explainer(user_input)
    st.write(cls_explainer.visualize())
    # st.divider()
    if cls_explainer.predicted_class_index == 1:
        st.subheader('Label Predicted: _Clickbait_')
    else:
        st.subheader('Label Predicted: _Not-Clickbait_')
