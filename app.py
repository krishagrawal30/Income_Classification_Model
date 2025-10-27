import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("income_model.pkl")

st.set_page_config(page_title="Income Classification", page_icon="💼", layout="centered")
st.title("💼 Income Classification App")
st.write("Predict whether a person earns **>50K or <=50K** based on census data.")

# --- Input fields ---
st.subheader("Enter User Information")

workclass = st.selectbox("Workclass", ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov","Local-gov", "State-gov", "Without-pay", "Never-worked"])

education = st.selectbox('Education', ['School', 'HS-grad', 'Bachelors', 'Masters', 'Doctorate', 'Some-college', 'Assoc-acdm', 'Assoc-voc', 'Prof-school'])

occupation = st.selectbox("Occupation", [
    "Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial",
    "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical",
    "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"
])

relationship = st.selectbox("Relationship", ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"])

race = st.selectbox('Race', ['White', 'Other'])

sex = st.selectbox("Sex", ["Male", "Female"])

native_country = st.selectbox("Native Country", [
    "United-States", "Cuba", "Jamaica", "India", "Mexico", "Puerto-Rico",
    "Honduras", "England", "Canada", "Germany", "Iran", "Philippines",
    "Poland", "Columbia", "Cambodia", "Thailand", "Ecuador", "Laos",
    "Taiwan", "Haiti", "Portugal", "Dominican-Republic", "El-Salvador",
    "France", "Guatemala", "Italy", "China", "Japan", "Yugoslavia",
    "Peru", "Outlying-US(Guam-USVI-etc)", "Scotland", "Trinadad&Tobago",
    "Greece", "Nicaragua", "Vietnam", "Hong", "Ireland", "Hungary",
    "Holand-Netherlands"
])

age = st.number_input("Age", min_value=17, max_value=90, value=30)
fnlwgt = st.number_input("Final Weight (fnlwgt)", min_value=10000, max_value=1000000, value=200000)
capital_gain = st.number_input("Capital Gain", min_value=0, max_value=100000, value=0)
capital_loss = st.number_input("Capital Loss", min_value=0, max_value=100000, value=0)
hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=100, value=40)

# --- Convert inputs to dataframe ---
input_data = pd.DataFrame({
    'workclass': [workclass],
    'education': [education],
    'occupation': [occupation],
    'relationship': [relationship],
    'race': [race],
    'sex': [sex],
    'native.country': [native_country],
    'age': [age],
    'fnlwgt': [fnlwgt],
    'capital.gain': [capital_gain],
    'capital.loss': [capital_loss],
    'hours.per.week': [hours_per_week]
})

# --- Preprocessing (One-Hot Encoding same as training) ---
input_encoded = pd.get_dummies(input_data)
# Align with training columns
train_columns = model.feature_names_in_
input_encoded = input_encoded.reindex(columns=train_columns, fill_value=0)

# --- Prediction ---
if st.button("Predict Income"):
    prediction = model.predict(input_encoded)[0]
    if prediction == '>50K':
        st.success("✅ Predicted Income: **>50K**")
    else:
        st.warning("💰 Predicted Income: **<=50K**")
