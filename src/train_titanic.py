import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

def preprocess(df):
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Sex'] = df['Sex'].map({'male' : 0, 'female' : 1})
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    target ='Survived'
    
    return df[features], df[target]

X, y = preprocess(df)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, max_depth = 5, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")

model_dir = '../models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    print(f"Created directory: {model_dir}")


joblib.dump(model, '../models/titanic_rf_model.joblib')
print("Model saved to models/titanic_rf_model.joblib")

print("\nFeature Importance:")
for name, importance in zip(X.columns, model.feature_importances_):
    print(f"{name}: {importance:.2%}")
