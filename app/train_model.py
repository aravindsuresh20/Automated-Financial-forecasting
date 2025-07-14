import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(base_dir, 'dataset', 'startup_funding.xlsx')

print(f"DEBUG: Attempting to load dataset from: {dataset_path}")
if not os.path.exists(dataset_path):
    print(f"ERROR: Dataset not found at: {dataset_path}", file=os.sys.stderr)
    exit(1)

df = pd.read_excel(dataset_path, engine='openpyxl')

# Normalize string fields
for col in ['Industry Vertical', 'SubVertical', 'City  Location', 'InvestmentnType']:
    df[col] = df[col].astype(str).str.strip().str.lower().fillna('unknown')

# Clean 'Amount in USD'
df['Amount in USD'] = df['Amount in USD'].astype(str).str.replace(',', '')
df['Amount in USD'] = pd.to_numeric(df['Amount in USD'], errors='coerce')
df = df.dropna(subset=['Amount in USD'])

features = ['Industry Vertical', 'SubVertical', 'City  Location', 'InvestmentnType']
X = df[features]
y = df['Amount in USD']

label_encoders = {}
for col in X.columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)

model_dir = os.path.join(base_dir, 'model')
os.makedirs(model_dir, exist_ok=True)
joblib.dump(label_encoders, os.path.join(model_dir, 'label_encoders.pkl'))
joblib.dump(model, os.path.join(model_dir, 'xgboost_model.pkl'))

print("✅ Model trained and saved.")
