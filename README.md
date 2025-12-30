# 😊 Emotion Text Classification

This project predicts emotions from text using **machine learning models**.  
It classifies emotions such as **Sadness, Joy, Love, Anger, Fear, and Surprise** and demonstrates **before and after hyperparameter tuning** results through a **Streamlit web application**.

---

## 🔧 Tech Stack

- **Programming Language:** Python
- **Libraries & Tools:** Pandas, Scikit-learn, TF-IDF Vectorizer, Naive Bayes, Logistic Regression, Streamlit, Hugging Face Datasets

---

## 📌 Features

- Loaded the **`air/emotion` dataset** from Hugging Face
- Combined train, validation, and test splits into a single dataset
- Preprocessed text using **TF-IDF Vectorizer**
- Trained and evaluated:
  - **Naive Bayes** (before & after tuning)
  - **Logistic Regression** (before & after tuning)
- Compared model **accuracy before and after tuning**
- Developed a **Streamlit UI** for real-time emotion prediction

---

## 🚀 How to Run

### 1️⃣ Clone the repository and navigate into it

```
git clone https://github.com/PranavSharma195/Emotion-Prediction.git
cd Emotion-Prediction
```

### 2️⃣ Install required packages

```
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit app

```
streamlit run app.py
```

## 📁 Project Structure

Emotion-Prediction/  
├── app.py # Streamlit application  
├── Emotion_Prediction.ipynb # Model training & analysis  
├── air_emotion_full_dataset.csv # Combined dataset  
├── nb_model_before.pkl # Naive Bayes before tuning  
├── nb_model_after.pkl # Naive Bayes after tuning  
├── lr_model_before.pkl # Logistic Regression before tuning  
├── lr_model_after.pkl # Logistic Regression after tuning  
├── tfidf_vectorizer.pkl # TF-IDF vectorizer  
├── X_test_tfidf.pkl # Test features  
├── y_test.pkl # Test labels  
├── Emotion_Prediction.pdf # Project report  
└── README.md # This documentation

---

## 📈 Future Improvements

- Use deep learning models (LSTM/BERT) for better accuracy
- Add prediction confidence scores
- Deploy online via Streamlit Cloud or Heroku

---

## 🤝 Contributions

- Feel free to fork the repository and open a pull request to improve the models or app.

---

## 📬 Contact

Created by **Pranav Sharma** – feel free to reach out!
