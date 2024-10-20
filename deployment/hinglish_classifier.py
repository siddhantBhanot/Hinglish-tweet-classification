import streamlit as st
from googletrans import Translator
from indictrans import Transliterator
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load your pre-trained SVM model
svm_model = joblib.load('svm_model.pkl')

# Load the vectorizer used for transforming data
vectorizer = joblib.load('vectorizer.pkl')  # Ensure you save and load the TF-IDF vectorizer

# Create a translator instance for Google Translate API
translator = Translator()

# Create an instance for the Transliterator
transliterator = Transliterator(source='eng', target='hin')

# Streamlit app header
st.title("Hinglish Text Classification")

st.write('**This model is based on my minor project:** [GitHub Repository](https://github.com/siddhantBhanot/Hinglish-tweet-classification/tree/master)')
st.write('It will classify your Hinglish text into three categories:')
st.write('**OAG** -> Overtly Aggressive')
st.write('**CAG** -> Covertly Aggressive')
st.write('**NAG** -> Not Aggressive')

# User input
user_input = st.text_input("Enter your Hinglish text:")

if st.button("Classify Text"):
    if user_input:
        # Step 1: Translate Roman Hinglish into Devanagari script using Google Translate API
        translated_text = translator.translate(user_input, src='en', dest='hi').text

        # Step 2: Transliterate Devanagari text into proper Hindi script using Indictrans
        transliterated_text = transliterator.transform(translated_text)

        # Step 3: Transform the text using the loaded TF-IDF vectorizer
        transformed_text = vectorizer.transform([transliterated_text])

        # Step 4: Predict the category using the SVM model
        prediction = svm_model.predict(transformed_text)

        # Output the prediction
        st.write(f"Prediction: {prediction[0]}")
    else:
        st.write("Please enter some text to classify.")
