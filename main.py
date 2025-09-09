import streamlit as st
import requests
import pandas as pd

# Set page config for better presentation
st.set_page_config(
    page_title="Sentiment Analysis Predictor",
    page_icon="🧠",
    layout="centered"
)

# Inject custom CSS for styling
st.markdown(
    """
    <style>
    /* Background and font */
    body {
        background-color: #f0f8ff;
        color: #034d4d;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    /* Button styling */
    div.stButton > button:first-child {
        background-color: #00796b;
        color: white;
        font-size: 16px;
        padding: 10px 25px;
        border-radius: 8px;
        transition: background-color 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #004d40;
        color: white;
    }
    /* Input and text area styling */
    textarea, input[type=text] {
        border: 2px solid #00796b;
        border-radius: 8px !important;
        padding: 10px !important;
    }
    /* File uploader style */
    .css-1r17p5d.edgvbvh3 {
        border: 2px solid #00796b !important;
        border-radius: 8px !important;
        padding: 10px !important;
        background-color: #e0f2f1 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title with emoji
st.title("🧠 Sentiment Analysis Predictor")

# Arrange options and inputs in columns
col1, col2 = st.columns([2,1])
with col1:
    model_choice = st.selectbox("Choose Model", ["Decision Tree", "Random Forest", "XGBoost"])
with col2:
    st.markdown("**Model:** " + model_choice)

model_map = {"Decision Tree": "dt", "Random Forest": "rf", "XGBoost": "xgb"}
selected_model = model_map[model_choice]

uploaded_file = st.file_uploader("Upload CSV for Bulk Prediction", type=["csv"], help="Upload a CSV file with a 'text' column for batch predictions")

text_input = st.text_area("Input Text for Single Prediction", height=150, placeholder="Enter your text here...")

API_URL = "http://127.0.0.1:5000"  # Make sure backend matches

if st.button("Predict Sentiment"):
    if text_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing..."):
            response = requests.post(f"{API_URL}/predict", json={"text": text_input, "model": selected_model})
            if response.status_code == 200:
                st.json(response.json())
            else:
                st.error("Error: Unable to get prediction from backend.")

if uploaded_file:
    if st.button("Predict Sentiment for CSV"):
        with st.spinner("Processing CSV..."):
            files = {"file": uploaded_file}
            data = {"model": selected_model}
            response = requests.post(f"{API_URL}/bulk_predict", files=files, data=data)
            if response.status_code == 200:
                results = response.json()
                df_results = pd.DataFrame(results)
                st.dataframe(df_results)
                csv_data = df_results.to_csv(index=False)
                st.download_button("Download Predictions CSV", csv_data, file_name="predictions.csv", mime="text/csv")
            else:
                st.error("Error: Unable to process CSV on backend.")
