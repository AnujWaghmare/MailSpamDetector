import streamlit as st
import pickle

# Load the pre-trained model and vectorizer
try:
    model = pickle.load(open('spam.pkl', 'rb'))
    cv = pickle.load(open('vectorizer.pkl', 'rb'))
except FileNotFoundError as e:
    st.error("Error loading model or vectorizer. Please ensure the 'spam.pkl' and 'vectorizer.pkl' files are present.")
    st.stop()

# Set up the Streamlit app
st.title("📧 Email Spam Classification Application")
st.write(
    """
    Welcome to the Email Spam Classifier!  
    This application uses Machine Learning to determine whether an email is **Spam** or **Not Spam (Ham)**.  
    """
)

# Input area for the user
st.subheader("Classification")
user_input = st.text_area("Enter the email text below for classification:", height=150)

# Classify button
if st.button("Classify"):
    if user_input.strip():  # Check if input is not empty
        # Preprocess and predict
        try:
            data = [user_input]
            vec = cv.transform(data).toarray()  # Transform input using the vectorizer
            result = model.predict(vec)  # Predict using the loaded model
            
            # Display the result
            if result[0] == 0:
                st.success("✅ This is NOT a Spam Email!")
            else:
                st.error("🚨 This is a SPAM Email!")
        except Exception as e:
            st.error(f"An error occurred during classification: {e}")
    else:
        st.warning("⚠️ Please enter email text before classifying.")
