import streamlit as st
import pickle
import string
import nltk

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

# Stemming function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
            
    return " ".join(y)

# Load the vectorizer and model
try:
    with open('vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"Error loading model or vectorizer: {e}")

st.title("Spam Message Verifier System")

input_msg = st.text_area("Enter your message below:")

if st.button('Click here to check'):
    if not input_msg:
        st.warning("Please enter a message to check.")
    else:
        # 1. Preprocess
        transformed_msg = transform_text(input_msg)
        
        # 2. Vectorize
        try:
            vector_input = tfidf.transform([transformed_msg])
        except Exception as e:
            st.error(f"Error transforming input: {e}")
            vector_input = None
        
        if vector_input is not None:
            # 3. Predict
            try:
                result = model.predict(vector_input)[0]
                
                # 4. Display
                if result == 1:
                    st.header("Spam")
                else:
                    st.header("Not Spam")
            except Exception as e:
                st.error(f"Error making prediction: {e}")
