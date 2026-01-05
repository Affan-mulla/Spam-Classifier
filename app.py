import json
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from streamlit_lottie import st_lottie
import time

# Load NLTK resources
nltk.download('stopwords')

# Create a PorterStemmer object
ps = PorterStemmer()

# Load the vectorizer and model from pickle files
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Function to load Lottie file
def load_lottiefile(filepath: str):
    with open(filepath, 'r') as f:
        return json.load(f)

# Function to transform text
def transform_text(text: str) -> str:
    text = text.lower()
    text = text.split()  # 🚀 no nltk tokenizer

    text = [word for word in text if word.isalnum()]
    text = [word for word in text if word not in stopwords.words('english')]
    text = [ps.stem(word) for word in text]

    return " ".join(text)

# App layout
st.image("logo.svg")
st.title("Identify spam messages with a click of a button")
st.markdown('Protect yourself from *getting spammed* by using this service.')

# Columns for layout
# col1, col2, col3 = st.columns((1.5, 0.5, 1))

# with col1:
#     st.header("How it works")
#     st.markdown("This application uses **Multinomial Naive Bayes** to classify messages into Spam or Not Spam.   "
#                 "*The **precision** of the result is 100% and the **accuracy** is 97.2%.*")

# with col2:
#     st.text("")

# with col3:
#     lottie_spam = load_lottiefile("side_image.json")
#     st_lottie(
#         lottie_spam,
#         speed=0.5,
#         reverse=False,
#         loop=True,
#         quality="high",
#         key=None,
#         height=250,
#         width=250,
#     )

# Input message area
st.header("Email/SMS Spam Classifier")
input_msg = st.text_area("Enter the message")

# Predict button
if st.button('Predict'):
    # Preprocess input message
    transformed_msg = transform_text(input_msg)
    # Vectorize input message
    vector_input = tfidf.transform([transformed_msg])
    # Predict spam or not spam
    result = model.predict(vector_input)[0]
    # Display result
    if result == 1:
        with st.spinner('Wait for it...'):
            time.sleep(1)
        st.error("This is a Spam message")
    else:
        with st.spinner('Wait for it...'):
            time.sleep(1)
        st.success("This is not a Spam Message")
