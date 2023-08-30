import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers_interpret import SequenceClassificationExplainer

# Define the function to load the model and tokenizer
@st.cache(allow_output_mutation=True)
def get_model_and_tokenizer():
    model_name = "samanjoy2/banglaclickbert_finetuned_sequence_classification_clickbait"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

# Load the tokenizer and model
tokenizer, model = get_model_and_tokenizer()

# Set page title and icon
st.set_page_config(page_title="Bangla Clickbait Detection", page_icon=":newspaper:")

# Sidebar with example headlines
with st.sidebar:
    st.image("your_logo.png", use_column_width=True)
    st.subheader('Example Headlines:')
    st.write("Clickbait Examples:")
    st.write("1. অবশেষে মুখ খুললেন রাজ, যা বললেন পরীমনি ও সন্তান প্রসঙ্গে")
    st.write("2. পান্তা ভাতের ইংরেজি কী জানেন? খুব সহজ! মাথায় আসছে না তো! জানুন")
    st.write("3. এক টানেই আধ কোটির ইলিশ! সাগরে জাল ফেলতেই ভাগ্য বদল মৎস্যজীবীর! ইলিশে ইলিশে ছেয়ে গেল বাজার!")
    st.write("Non-Clickbait Examples:")
    st.write("1. যুক্তরাজ্যে ফ্লাইট বিপর্যয়, ভোগান্তি থাকবে ‘কয়েকদিন’")
    st.write("2. চট্টগ্রামে খাল–নালায় মানুষ মরছে, তবু নিরাপত্তাবেষ্টনী উঠছে না")
    st.write("3. বায়ুদূষণে বাংলাদেশের মানুষের গড় আয়ু কমছে প্রায় ৭ বছর")

# Main content section
st.title('Bangla Clickbait Detection :newspaper:')
st.header('Check if a Bangla News Headline is Clickbait')

# User input and analysis button
user_input = st.text_area('Enter Text to Analyze')
analyze_button = st.button("Analyze")

# Perform analysis if input and button are clicked
if user_input and analyze_button:
    cls_explainer = SequenceClassificationExplainer(model, tokenizer)
    word_attributions = cls_explainer(user_input)
    
    # Visualize interpretation
    st.write(cls_explainer.visualize())
    
    # Display prediction label
    predicted_label = "Clickbait" if cls_explainer.predicted_class_index == 1 else "Not-Clickbait"
    st.subheader(f'Predicted Label: _{predicted_label}_')
