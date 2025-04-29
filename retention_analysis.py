import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# load and prep data
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
df['Churn'] = df['Churn'].map({'Yes':1, 'No':0})

# features
features = ['tenure','MonthlyCharges','TotalCharges','SeniorCitizen','Contract','OnlineSecurity']
X = pd.get_dummies(df[features], drop_first=True)
y = df['Churn']

# RF model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print(f"Accuracy: {model.score(X_test, y_test)}")

# feature importance plot
fi = pd.DataFrame({'feature':X.columns, 'importance':model.feature_importances_}).sort_values('importance')
plt.barh(fi['feature'], fi['importance'], color='gray')
plt.title('Ranked Churn Predictors');
plt.tight_layout()
plt.show()