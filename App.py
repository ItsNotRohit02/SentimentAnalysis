import streamlit as st
from transformers import AutoTokenizer as AT
from transformers import AutoModelForSequenceClassification as AM
import torch

st.set_page_config(page_title="Sentiment Analysis", page_icon=":smile:")

with st.container():
    st.title('Basic Sentiment Analysis :grin: :sob: :expressionless:')
    st.write("This is a very basic Sentiment Analysis program that returns a value between -1 and 1 depending on the "
             "text entered. The program makes use of basic Natural Language Processing Algorithms.")
    st.write("---")

with st.container():
    tokenizer = AT.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    model = AM.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    text = st.text_area('Enter Text Here', value='')
    tokens = tokenizer.encode(text, return_tensors='pt')
    result = model(tokens)
    value = int(torch.argmax(result.logits)) + 1

    if value == 1:
        st.error('The text entered is Extremely Negative')
    elif value == 2:
        st.error('The text entered is Negative')
    elif value == 3:
        st.info('The text entered is Neutral')
    elif value == 4:
        st.success('The text entered is Positive')
    elif value == 5:
        st.success('The text entered is Overwhelmingly Positive')
    st.write(value)
    st.caption('Made by Rohit')
    st.markdown("[![Foo](https://cdn-icons-png.flaticon.com/24/25/25231.png)](https://github.com/ItsNotRohit02)")






