import streamlit as st
from transformers_interpret import SequenceClassificationExplainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

@st.cache(allow_output_mutation=True)
def get_model():
    model_name = "samanjoy2/banglaclickbert_finetuned_sequence_classification_clickbait"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = get_model()

st.title('Bangla Clickbait Detection :sunglasses:')
st.header('Check whether a Bangla News Headline is Clickbait or not')

col1, col2 = st.columns(2)

with col1:
    st.subheader('Some Examples of Clickbait Headlines:')
    clickbait_examples = [
        "অবশেষে মুখ খুললেন রাজ, যা বললেন পরীমনি ও সন্তান প্রসঙ্গে",
        "পান্তা ভাতের ইংরেজি কী জানেন? খুব সহজ! মাথায় আসছে না তো! জানুন",
        "এক টানেই আধ কোটির ইলিশ! সাগরে জাল ফেলতেই ভাগ্য বদল মৎস্যজীবীর! ইলিশে ইলিশে ছেয়ে গেল বাজার!"
    ]
    selected_clickbait = st.selectbox("Select Clickbait Example", clickbait_examples)
    copy_clickbait = st.button("Copy Clickbait Example")

with col1:
    st.subheader('Some Examples of Non-Clickbait Headlines:')
    non_clickbait_examples = [
        "যুক্তরাজ্যে ফ্লাইট বিপর্যয়, ভোগান্তি থাকবে ‘কয়েকদিন’",
        "চট্টগ্রামে খাল–নালায় মানুষ মরছে, তবু নিরাপত্তাবেষ্টনী উঠছে না",
        "বায়ুদূষণে বাংলাদেশের মানুষের গড় আয়ু কমছে প্রায় ৭ বছর"
    ]
    selected_non_clickbait = st.selectbox("Select Non-Clickbait Example", non_clickbait_examples)
    copy_non_clickbait = st.button("Copy Non-Clickbait Example")

user_input = col2.text_area('Text to analyze')
button = col2.button("Analyze")

if copy_clickbait:
    user_input = col2.text_area('Text to analyze', selected_clickbait)

if copy_non_clickbait:
    user_input = col2.text_area('Text to analyze', selected_non_clickbait)

if user_input and button:
    cls_explainer = SequenceClassificationExplainer(model, tokenizer)
    word_attributions = cls_explainer(user_input)
    st.write(cls_explainer.visualize())
    if cls_explainer.predicted_class_index == 1:
        st.subheader('Label Predicted: _Clickbait_')
    else:
        st.subheader('Label Predicted: _Not-Clickbait_')
