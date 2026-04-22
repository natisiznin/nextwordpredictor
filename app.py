import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Cache the model and tokenizer to prevent reloading on every UI interaction
@st.cache_resource
def load_ml_artifacts():
    model = load_model('next_word_lstm.keras')
    with open('Tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, Tokenizer = load_ml_artifacts()

# 2. Your prediction function
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# 3. Streamlit Application UI
st.title("Next Word Prediction Engine")
st.write("Type a few words below, and the LSTM model will guess what comes next.")

# IMPORTANT: You must set this to the exact max_sequence_len you used during training!
# For example, if you trained on n-grams up to length 10, set this to 10.
MAX_SEQUENCE_LEN = 10 

# User Input
input_text = st.text_input("Enter your starting text:", placeholder="e.g., The cat sat on the...")

# Prediction Trigger
if st.button("Predict"):
    if input_text.strip() == "":
        st.warning("Please enter some text to get a prediction.")
    else:
        with st.spinner("Analyzing sequence..."):
            predicted_word = predict_next_word(model, Tokenizer, input_text, MAX_SEQUENCE_LEN)
            
            if predicted_word:
                st.success(f"**Predicted Word:** {predicted_word}")
                st.info(f"**Combined:** {input_text} {predicted_word}")
            else:
                st.error("The model couldn't find a confident prediction for this text.")