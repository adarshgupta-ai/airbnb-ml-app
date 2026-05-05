import nltk

# Download tokenizer if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

import streamlit as st
from model import predict_price, get_sentiment

st.set_page_config(page_title="Airbnb ML App", layout="wide")

st.title("🏠 Airbnb Price Prediction + Sentiment Analysis")

# Sidebar
st.sidebar.header("Input Features")

min_nights = st.sidebar.slider("Minimum Nights", 1, 30, 2)
reviews = st.sidebar.slider("Number of Reviews", 0, 500, 50)
reviews_per_month = st.sidebar.slider("Reviews per Month", 0.0, 10.0, 1.0)

location = st.sidebar.selectbox(
    "Location",
    ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
)

# 🔥 Model selection
model_choice = st.sidebar.selectbox(
    "Choose Model",
    ["Linear Regression", "Random Forest", "Ensemble"]
)

review_text = st.text_area("💬 Enter Guest Review (optional)")

# Predict
if st.button("Predict Price"):

    features = {
        "minimum_nights": min_nights,
        "number_of_reviews": reviews,
        "reviews_per_month": reviews_per_month,
        "neighbourhood_group": location
    }

    price = predict_price(model_choice, features)

    sentiment = get_sentiment(review_text)

    # Hybrid logic (price adjustment)
    if review_text.strip():
        if sentiment > 0.3:
            price *= 1.08
        elif sentiment < -0.3:
            price *= 0.90

    col1, col2 = st.columns(2)

    with col1:
        st.metric("💰 Predicted Price (USD)", f"${round(price, 2)}")

    with col2:
        if not review_text.strip():
            st.info("😐 No review provided")
        elif sentiment > 0.3:
            st.success("😊 Positive Review")
        elif sentiment < -0.3:
            st.error("😡 Negative Review")
        else:
            st.info("😐 Neutral Review")

    st.write(f"Sentiment Score: {round(sentiment, 2)}")

    st.caption("Hybrid ML + NLP system with Ensemble & Optimization")