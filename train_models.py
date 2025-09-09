import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os


def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text).strip()
    return text


# Load training CSV
df = pd.read_csv('imdb_train.csv')


df['clean_text'] = df['Sentence'].apply(preprocess)

# Using CountVectorizer with max_features to limit vocab size for efficiency
vectorizer = CountVectorizer(max_features=10000)
X = vectorizer.fit_transform(df['clean_text'])
# Use MaxAbsScaler which supports sparse data and avoids memory error
scaler = MaxAbsScaler()
X_scaled = scaler.fit_transform(X)  # No toarray() conversion required

y = df['Sentiment']

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train decision tree classifier
model_dt = DecisionTreeClassifier(random_state=42)
model_dt.fit(X_train, y_train)

# Train random forest classifier
model_rf = RandomForestClassifier(random_state=42)
model_rf.fit(X_train, y_train)

# Train XGBoost classifier
model_xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model_xgb.fit(X_train, y_train)

# Ensure 'models' directory exists
os.makedirs('models', exist_ok=True)

# Save vectorizer, scaler, and models
joblib.dump(vectorizer, 'models/countVectorizer.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(model_dt, 'models/model_dt.pkl')
joblib.dump(model_rf, 'models/model_rf.pkl')
joblib.dump(model_xgb, 'models/model_xgb.pkl')

print("Models trained and saved!")

# Evaluate models accuracy and print classification reports
for model, name in zip([model_dt, model_rf, model_xgb], ['Decision Tree', 'Random Forest', 'XGBoost']):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
