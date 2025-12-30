import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

# ---------------------------
# Small heading
# ---------------------------
st.markdown("<h6 style='text-align: left; color: gray;'>Pranav Sharma Text Classifier</h6>", unsafe_allow_html=True)
st.title("Emotion Prediction Before & After Hyperparameter Tuning")

# ---------------------------
# Load models and vectorizer
# ---------------------------
nb_model_before = joblib.load("nb_model_before.pkl")
nb_model_after = joblib.load("nb_model_after.pkl")
lr_model_before = joblib.load("lr_model_before.pkl")
lr_model_after = joblib.load("lr_model_after.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
X_test_tfidf = joblib.load("X_test_tfidf.pkl")
y_test = joblib.load("y_test.pkl")

# ---------------------------
# Emotion mapping
# ---------------------------
emotion_map = {
    0: "Sadness",
    1: "Joy",
    2: "Love",
    3: "Anger",
    4: "Fear",
    5: "Surprise"
}

# ---------------------------
# Helper function for card
# ---------------------------
def create_card(title, content, bg_color="#FFF3CD"):
    st.markdown(
        f"<div style='background-color:{bg_color}; color:black; padding:20px; border-radius:15px; box-shadow:2px 2px 10px gray;'>"
        f"<strong>{title}</strong><br>{content}</div>",
        unsafe_allow_html=True
    )
    st.write("")  # space after card

# ---------------------------
# Naive Bayes Section
# ---------------------------
st.header("Naive Bayes")
nb_input = st.text_area("Enter text for Naive Bayes prediction:")

if st.button("Predict Naive Bayes"):
    vec = vectorizer.transform([nb_input])

    # Before Tuning
    nb_before_pred = emotion_map[nb_model_before.predict(vec).item()]
    nb_before_acc = accuracy_score(y_test, nb_model_before.predict(X_test_tfidf))
    create_card("Before Tuning Prediction", nb_before_pred, "#FFF3CD")
    create_card("Before Tuning Accuracy", f"{nb_before_acc:.4f}", "#FFF3CD")

    # After Tuning
    nb_after_pred = emotion_map[nb_model_after.predict(vec).item()]
    nb_after_acc = accuracy_score(y_test, nb_model_after.predict(X_test_tfidf))
    create_card("After Tuning Prediction", nb_after_pred, "#D4EDDA")
    create_card("After Tuning Accuracy", f"{nb_after_acc:.4f}", "#D4EDDA")

# ---------------------------
# Logistic Regression Section
# ---------------------------
st.header("Logistic Regression")
lr_input = st.text_area("Enter text for Logistic Regression prediction:")

if st.button("Predict Logistic Regression"):
    vec = vectorizer.transform([lr_input])

    # Before Tuning
    lr_before_pred = emotion_map[lr_model_before.predict(vec).item()]
    lr_before_acc = accuracy_score(y_test, lr_model_before.predict(X_test_tfidf))
    create_card("Before Tuning Prediction", lr_before_pred, "#FFF3CD")
    create_card("Before Tuning Accuracy", f"{lr_before_acc:.4f}", "#FFF3CD")

    # After Tuning
    lr_after_pred = emotion_map[lr_model_after.predict(vec).item()]
    lr_after_acc = accuracy_score(y_test, lr_model_after.predict(X_test_tfidf))
    create_card("After Tuning Prediction", lr_after_pred, "#D4EDDA")
    create_card("After Tuning Accuracy", f"{lr_after_acc:.4f}", "#D4EDDA")

# ---------------------------
# Accuracy Comparison Section (bottom)
# ---------------------------
st.header("Model Accuracy Comparison")
results = {
    "Naive Bayes": {
        "Before": accuracy_score(y_test, nb_model_before.predict(X_test_tfidf)),
        "After": accuracy_score(y_test, nb_model_after.predict(X_test_tfidf))
    },
    "Logistic Regression": {
        "Before": accuracy_score(y_test, lr_model_before.predict(X_test_tfidf)),
        "After": accuracy_score(y_test, lr_model_after.predict(X_test_tfidf))
    }
}
accuracy_df = pd.DataFrame(results).T
st.subheader("📊 Accuracy Before vs After Hyperparameter Tuning")
st.dataframe(accuracy_df, use_container_width=True)
st.bar_chart(accuracy_df)
