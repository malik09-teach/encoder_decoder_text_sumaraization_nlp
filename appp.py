import streamlit as st
import numpy as np
import pickle
import re
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==========================================
# 1. LOAD YOUR SAVED FILES
# ==========================================
@st.cache_resource # This tells Streamlit to only load the heavy model once!
def load_ai_components():
    # Load tokenizers
    with open('x_tokenizer.pkl', 'rb') as f:
        x_tokenizer = pickle.load(f)
    with open('y_tokenizer.pkl', 'rb') as f:
        y_tokenizer = pickle.load(f)
        
    # Load the main trained model
    model = load_model('best_model.keras')
    
    # --- SPLIT INTO INFERENCE MODELS HERE ---
    # (You will paste your encoder_model and decoder_model code here, 
    # pulling the layers directly from the loaded 'model')
    
    return x_tokenizer, y_tokenizer, encoder_model, decoder_model

x_tokenizer, y_tokenizer, encoder_model, decoder_model = load_ai_components()

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
# Paste your exact clean_text() regex function from Colab here!
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Paste your decode_sequence() loop from Colab here!
def generate_summary(input_sequence):
    # Your step-by-step prediction loop goes here
    pass 

# ==========================================
# 3. THE STREAMLIT WEBSITE UI
# ==========================================
st.title("📰 AI Article Summarizer")
st.write("Paste a news article below, and my Seq2Seq Neural Network will summarize it!")

# Create a text box for the user to paste an article
user_article = st.text_area("Input Article:", height=300)

# Create a "Summarize" button
if st.button("Summarize!"):
    if user_article:
        with st.spinner("The AI is reading the article..."):
            # 1. Clean the user's text
            cleaned_text = clean_text(user_article)
            
            # 2. Convert to numbers and pad to 400
            seq = x_tokenizer.texts_to_sequences([cleaned_text])
            padded_seq = pad_sequences(seq, maxlen=400, padding='post', truncating='post')
            
            # 3. Generate the summary!
            final_summary = generate_summary(padded_seq)
            
            # 4. Display the result on the website
            st.success("Summary Generated!")
            st.write(final_summary)
    else:
        st.warning("Please paste an article first!")