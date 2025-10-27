import streamlit as st
import pandas as pd
import joblib
from PIL import Image

st.set_page_config(
    page_title="Income Classification App",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

model = joblib.load("income_model.pkl")

st.title(" Income Classification App")
st.markdown(
    """
    <style>
    .main-title {
        font-size:30px !important;
        color:#1f77b4;
        font-weight:600;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("### Predict whether a person earns **>50K** or **≤50K** based on U.S. Census data.")

st.divider()

st.sidebar.header("📊 About the Project")
st.sidebar.write(
    """
    This ML model predicts whether an individual's income exceeds 50K USD
    using demographic and occupational data.
    """
)
st.sidebar.info(
    """
    **Model Used:** Gradient Boosting Classifier  
    **Best Accuracy:** 87.1%  
    **Dataset:** UCI Adult Census  
    """
)
st.sidebar.write("👨‍💻 *Developed by Krish Agrawal*")

st.subheader("🧾 Enter User Information")

col1, col2 = st.columns(2)

with col1:
    workclass = st.selectbox(
        "Workclass",
        ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov",
         "State-gov", "Without-pay", "Never-worked"]
    )

    education = st.selectbox(
        "Education",
        ["School", "HS-grad", "Bachelors", "Masters", "Doctorate", "Some-college",
         "Assoc-acdm", "Assoc-voc", "Prof-school"]
    )

    occupation = st.selectbox(
        "Occupation",
        ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial",
         "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical",
         "Farming-fishing", "Transport-moving", "Priv-house-serv",
         "Protective-serv", "Armed-Forces"]
    )

    relationship = st.selectbox(
        "Relationship",
        ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"]
    )

    race = st.selectbox("Race", ["White", "Other"])
    sex = st.radio("Sex", ["Male", "Female"])

with col2:
    native_country = st.selectbox(
        "Native Country",
        ["United-States", "India", "Mexico", "Philippines", "Germany", "Canada", "China",
         "England", "France", "Japan", "Vietnam", "Iran", "Greece", "Italy", "Ireland",
         "Poland", "Portugal", "Cuba", "Jamaica", "Other"]
    )

    age = st.slider("Age", 17, 90, 30)
    fnlwgt = st.number_input("Final Weight (fnlwgt)", 10000, 1000000, 200000, step=1000)
    capital_gain = st.number_input("Capital Gain", 0, 100000, 0, step=500)
    capital_loss = st.number_input("Capital Loss", 0, 100000, 0, step=500)
    hours_per_week = st.slider("Hours per Week", 1, 100, 40)

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

input_encoded = pd.get_dummies(input_data)
train_columns = model.feature_names_in_
input_encoded = input_encoded.reindex(columns=train_columns, fill_value=0)

if st.button("Predict Income"):
    with st.spinner("Analyzing data and predicting..."):
        prediction = model.predict(input_encoded)[0]

    st.subheader("📈 Prediction Result:")
    if prediction == ">50K":
        st.success(" The model predicts that the person's income is **>50K USD**.")
        st.balloons()
    else:
        st.warning(" The model predicts that the person's income is **≤50K USD**.")

    st.info("Confidence: This prediction is based on your demographic and occupational details.")

st.divider()
st.caption("© 2025 Income Classification Model | Developed by Krish Agrawal 🚀")
