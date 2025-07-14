import pandas as pd
import joblib
import os
import sys

def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'model', 'xgboost_model.pkl')
    encoder_path = os.path.join(base_dir, 'model', 'label_encoders.pkl')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Encoder not found: {encoder_path}")

    model = joblib.load(model_path)
    encoders = joblib.load(encoder_path)
    return model, encoders

def preprocess_input(input_data, encoders):
    processed = {}

    for col, value in input_data.items():
        value = str(value).strip().lower()
        encoder = encoders.get(col)

        if encoder:
            class_list = [cls.lower() for cls in encoder.classes_]

            if value in class_list:
                original_class = encoder.classes_[class_list.index(value)]
                processed[col] = encoder.transform([original_class])[0]
            elif 'unknown' in class_list:
                processed[col] = encoder.transform(['Unknown'])[0]
                print(f"WARNING: Unknown value '{value}' for {col}. Mapped to 'Unknown'", file=sys.stderr)
            else:
                processed[col] = encoder.transform([encoder.classes_[0]])[0]
        else:
            raise ValueError(f"Missing encoder for column: {col}")

    expected_order = ['Industry Vertical', 'SubVertical', 'City  Location', 'InvestmentnType']
    processed_df = pd.DataFrame([processed]).reindex(columns=expected_order, fill_value=0)

    return processed_df

def predict_cost(model, input_df):
    return model.predict(input_df)[0]
