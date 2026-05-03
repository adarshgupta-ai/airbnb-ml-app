import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from textblob import TextBlob

# Load dataset
data = pd.read_csv("airbnb.csv")

# Clean price
data['price'] = data['price'].astype(str).str.replace('$','').str.replace(',','')
data['price'] = data['price'].astype(float)

# Select features
data = data[['price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'neighbourhood_group']]
data = data.dropna()

# Encode location
data = pd.get_dummies(data, columns=['neighbourhood_group'])

# Features & target
X = data.drop('price', axis=1)
y = np.log1p(data['price'])  # log transform

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Models
lr_model = LinearRegression()
rf_model = RandomForestRegressor(n_estimators=150, random_state=42)

lr_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# Prediction function
def predict_price(model_type, features_dict):

    df = pd.DataFrame([features_dict])
    df = pd.get_dummies(df)
    df = df.reindex(columns=X.columns, fill_value=0)

    if model_type == "Linear Regression":
        pred = lr_model.predict(df)[0]
    else:
        pred = rf_model.predict(df)[0]

    return np.expm1(pred)  # convert back

# Sentiment function
def get_sentiment(review):
    return TextBlob(review).sentiment.polarity