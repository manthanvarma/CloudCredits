import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.model_selection import train_test_split

# Load model and scaler
model = joblib.load('knn_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🩺 Diabetes Prediction App (KNN Model)")

# Input form
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
user_input = []

st.subheader("🔢 Enter Patient Details")
for f in features:
    val = st.number_input(f"Enter {f}:", min_value=0.0, step=0.1)
    user_input.append(val)

if st.button("🧠 Predict"):
    input_array = np.array(user_input).reshape(1, -1)
    scaled_input = scaler.transform(input_array)
    prediction = model.predict(scaled_input)[0]
    proba = model.predict_proba(scaled_input)[0][1]

    result = "Diabetic" if prediction == 1 else "Not Diabetic"
    st.success(f"Prediction: {result}")
    st.info(f"Confidence: {proba*100:.2f}%")

    # Bar chart of user input
    st.subheader("📊 Your Input Summary")
    fig1, ax1 = plt.subplots()
    ax1.bar(features, user_input, color='skyblue')
    ax1.set_ylabel("Values")
    ax1.set_title("Entered Feature Values")
    plt.xticks(rotation=45)
    st.pyplot(fig1)

# Compare with dataset averages
if st.checkbox("📉 Compare with Diabetic and Non-Diabetic Averages"):
    df = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv", header=None)
    df.columns = features + ['Outcome']
    diabetic_avg = df[df['Outcome'] == 1].mean()
    nondiabetic_avg = df[df['Outcome'] == 0].mean()

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    x = np.arange(len(features))
    width = 0.25

    ax2.bar(x - width, [nondiabetic_avg[f] for f in features], width, label='Non-Diabetic', color='green')
    ax2.bar(x, [diabetic_avg[f] for f in features], width, label='Diabetic', color='red')
    ax2.bar(x + width, user_input, width, label='You', color='blue')

    ax2.set_xticks(x)
    ax2.set_xticklabels(features, rotation=45)
    ax2.set_title("🔍 Feature Comparison")
    ax2.legend()
    st.pyplot(fig2)

# Confusion Matrix toggle
if st.checkbox("🧮 Show Confusion Matrix"):
    df = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv", header=None)
    df.columns = features + ['Outcome']
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_scaled = scaler.fit_transform(X)
    _, X_test, _, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)
    y_pred = model.predict(X_test)

    fig3, ax3 = plt.subplots()
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(ax=ax3)
    st.pyplot(fig3)


