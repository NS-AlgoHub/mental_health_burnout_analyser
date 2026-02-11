import pandas as pd
import pickle, os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

os.makedirs('model', exist_ok=True)
df = pd.read_csv('data/burnout_data.csv')

X = df.drop('burnout_risk', axis=1)
y = df['burnout_risk']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(random_state=42)
model.fit(X_scaled, y)

with open('model/burnout_model.pkl','wb') as f:
    pickle.dump((model, scaler), f)

print('Burnout prediction model trained')
