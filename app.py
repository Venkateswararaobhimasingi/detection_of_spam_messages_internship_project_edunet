import re
import pickle
import string
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem.porter import PorterStemmer
import nltk

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize components
ps = PorterStemmer()
tokenizer = TreebankWordTokenizer()
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Whitelist of non-spam URLs
whitelist = ['google.com', 'facebook.com']

# Streamlit app title
st.title("Email/SMS Spam Classifier")

# URL extraction and whitelist check
def extract_urls(text):
    url_pattern = re.compile(r'https?://[A-Za-z0-9./]+')
    return url_pattern.findall(text)

def is_url_whitelisted(url, whitelist):
    return any(domain in url for domain in whitelist)

# Text preprocessing function
def transform_text(text):
    text = text.lower()
    text = tokenizer.tokenize(text)
    text = ["<URL>" if "http" in i or "www" in i else i for i in text]
    text = [i for i in text if i.isalnum() or i == "<URL>"]
    text = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]
    return " ".join([ps.stem(i) for i in text])

# Streamlit input and prediction
input_sms = st.text_area("Enter the message")
if st.button('Predict'):
    urls = extract_urls(input_sms)
    is_ham_due_to_whitelist = any(is_url_whitelisted(url, whitelist) for url in urls)
    
    if is_ham_due_to_whitelist:
        st.header("Not Spam (Whitelisted URL detected)")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]
        
        if result == 1 or urls:
            st.header("Spam")
        else:
            st.header("Not Spam")
