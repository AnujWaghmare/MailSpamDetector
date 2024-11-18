
# Email Spam Classification Application

## 📧 About the Project
This is a **Machine Learning application** built with Python and Streamlit that classifies emails as **spam** or **ham (not spam)**. It leverages a trained ML model to analyze text input and predict whether the email is spam or not.

---

## ✨ Features
- **Interactive User Interface**: Enter an email and instantly classify it as spam or not.
- **Streamlit-Powered**: Provides a seamless and responsive web interface.
- **Pre-Trained ML Model**: Uses a Naive Bayes classifier for efficient and accurate predictions.
- **Custom Vectorization**: Email text is preprocessed with a TF-IDF vectorizer to feed the model.

---

## 🚀 How to Run
### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/Email-Spam-Classification.git
cd Email-Spam-Classification
```

### Step 2: Install Dependencies
Ensure you have Python installed. Run the following to install the required libraries:
```bash
pip install -r requirements.txt
```

### Step 3: Start the Application
Run the Streamlit app:
```bash
streamlit run app.py
```

### Step 4: Classify Emails
- Enter email text in the text box provided.
- Click **Classify** to check if the email is spam or not.

---

## 📂 Repository Structure
```plaintext
Email-Spam-Classification/
├── app.py                 # Main application script
├── spam.pkl               # Pre-trained model
├── vectorizer.pkl         # TF-IDF Vectorizer
├── requirements.txt       # List of dependencies
└── README.md              # Project documentation
```

---

## 🤝 Connect with Me
- [LinkedIn](https://www.linkedin.com/in/your-profile-name)
- [GitHub](https://github.com/your-github-username)

---

## 🛡️ License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
