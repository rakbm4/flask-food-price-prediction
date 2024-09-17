from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
import gdown

app = Flask(__name__)

# Path where the model will be saved
model_path = './rf_model.pkl'
# Google Drive shareable link for the model (you provided)
gdrive_link = 'https://drive.google.com/uc?id=1pDTowstt0Lnoz9REEyfMQb5u615zmfFt'

# Download model from Google Drive if it doesn't exist
if not os.path.exists(model_path):
    print("Model not found locally. Downloading from Google Drive...")
    gdown.download(gdrive_link, model_path, quiet=False)

# Load the trained Random Forest model and the encoded dataset
rf_model = joblib.load(model_path)
encoded_dataset_path = './data/encoded_wfp_food_prices_ind.csv'  # Update with correct path to your dataset

# Ensure that the dataset exists
if not os.path.exists(encoded_dataset_path):
    raise FileNotFoundError(f"Dataset not found at {encoded_dataset_path}")

encoded_dataset = pd.read_csv(encoded_dataset_path)

# Extract unique provinces, categories, and commodities from the one-hot encoded columns
provinces = [col.replace('province_', '') for col in encoded_dataset.columns if col.startswith('province_')]
categories = [col.replace('category_', '') for col in encoded_dataset.columns if col.startswith('category_')]
commodities = [col.replace('commodity_', '') for col in encoded_dataset.columns if col.startswith('commodity_')]

# Feature columns from the encoded dataset
feature_columns = list(encoded_dataset.drop('price(INR)', axis=1).columns)

@app.route('/')
def home():
    return render_template('index.html', provinces=provinces, categories=categories, commodities=commodities)

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    year = int(request.form['year'])
    month = int(request.form['month'])
    province = request.form['province']
    category = request.form['category']
    commodity = request.form['commodity']

    # Create a dictionary for the input data
    input_data = {
        'year': year,
        'month': month,
        f'province_{province}': 1,
        f'category_{category}': 1,
        f'commodity_{commodity}': 1
    }

    # Ensure all feature columns are present (set missing features to 0)
    for column in feature_columns:
        if column not in input_data:
            input_data[column] = 0

    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Reorder the input columns to match the training set
    input_df = input_df[feature_columns]

    # Make the prediction using the loaded model
    predicted_price = rf_model.predict(input_df)[0]

    # Render the result on the page
    return render_template('index.html', prediction_text=f'Predicted Price: {predicted_price:.2f} INR', 
                           provinces=provinces, categories=categories, commodities=commodities)

# EDA Page Route
@app.route('/eda')
def eda():
    return render_template('eda.html')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=True)
