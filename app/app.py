from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import pandas as pd
import sys
import re
from utils import load_model, preprocess_input, predict_cost

app = Flask(__name__)

# Constants
USD_TO_INR = 83.2
base_dir = os.path.dirname(os.path.abspath(__file__))
model, label_encoders = load_model()

# --- Helper: Normalization function for input and dataset ---
def normalize(text):
    """Lowercase, remove hyphens, punctuation, and whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[\-\s]+', '', text)      # remove hyphens and spaces
    text = re.sub(r'[^\w]', '', text)        # remove non-alphanumeric
    return text

# Load the original dataset once and normalize
dataset_path = os.path.join(base_dir, 'dataset', 'startup_funding.xlsx')
original_df = pd.read_excel(dataset_path, engine='openpyxl')

for col in ['Industry Vertical', 'SubVertical', 'City  Location']:
    original_df[col] = original_df[col].astype(str).apply(normalize)

original_df['Amount in USD'] = original_df['Amount in USD'].astype(str).str.replace(',', '').str.strip()
original_df['Amount in USD'] = pd.to_numeric(original_df['Amount in USD'], errors='coerce')

# Output path
DOWNLOAD_FOLDER = os.path.join(base_dir, 'output')
DOWNLOAD_FILE_NAME = 'prediction_results.xlsx'
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

@app.route('/')
def homepage():
    return render_template('homepage.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_page():
    if request.method == 'POST':
        city = request.form['city'].strip()
        industry = request.form['industry'].strip()
        subvertical = request.form['subvertical'].strip()

        input_data = {
            'City  Location': city,
            'Industry Vertical': industry,
            'SubVertical': subvertical,
            'InvestmentnType': 'Unknown'
        }

        # Normalize user input for matching
        norm_city = normalize(city)
        norm_industry = normalize(industry)
        norm_subvertical = normalize(subvertical)

        # Attempt to match from dataset
        match = original_df[
            (original_df['City  Location'] == norm_city) &
            (original_df['Industry Vertical'] == norm_industry) &
            (original_df['SubVertical'] == norm_subvertical)
        ]

        if not match.empty:
            prediction_usd = match.iloc[0]['Amount in USD']
            prediction_inr = round(prediction_usd * USD_TO_INR, 2)
            print("✅ Matched input to dataset. Using actual amount.", file=sys.stderr)
        else:
            processed_data = preprocess_input(input_data, label_encoders)
            prediction_usd = predict_cost(model, processed_data)
            prediction_inr = round(prediction_usd * USD_TO_INR, 2)
            print("⚠️ No match found. Using model prediction.", file=sys.stderr)

        # Save result to Excel
        output_path = os.path.join(DOWNLOAD_FOLDER, DOWNLOAD_FILE_NAME)
        output_columns = ['City  Location', 'Industry Vertical', 'SubVertical',
                          'Predicted Operational Cost (USD)', 'Predicted Operational Cost (INR)']

        new_row = {
            'City  Location': city,
            'Industry Vertical': industry,
            'SubVertical': subvertical,
            'Predicted Operational Cost (USD)': round(prediction_usd, 2),
            'Predicted Operational Cost (INR)': prediction_inr
        }
        new_df = pd.DataFrame([new_row], columns=output_columns)

        if os.path.exists(output_path):
            existing_df = pd.read_excel(output_path)
            existing_df = existing_df.reindex(columns=output_columns)
            updated_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            updated_df = new_df

        updated_df.to_excel(output_path, index=False)

        return redirect(url_for('result_page',
                                prediction_usd=round(prediction_usd, 2),
                                prediction_inr=prediction_inr,
                                city=city,
                                industry=industry,
                                subvertical=subvertical))

    return render_template('index.html')

@app.route('/result')
def result_page():
    return render_template('result.html',
                           prediction_usd=request.args.get('prediction_usd', 'N/A'),
                           prediction_inr=request.args.get('prediction_inr', 'N/A'),
                           city=request.args.get('city', 'N/A'),
                           industry=request.args.get('industry', 'N/A'),
                           subvertical=request.args.get('subvertical', 'N/A'))

@app.route('/download_output_file')
def download_output_file():
    output_path = os.path.join(DOWNLOAD_FOLDER, DOWNLOAD_FILE_NAME)
    if not os.path.exists(output_path):
        return "Prediction file not found.", 404
    return send_from_directory(DOWNLOAD_FOLDER, DOWNLOAD_FILE_NAME, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
