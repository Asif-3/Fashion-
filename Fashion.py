import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

df = pd.read_csv("fashion_dataset.csv") 
df.drop(columns=['id', 'productDisplayName', 'image'], inplace=True)
df.ffill(inplace=True)
print("\nTarget Class Distribution:")
print(df['masterCategory'].value_counts())
class_counts = df['masterCategory'].value_counts()
if any(class_counts < 2):
    print("start")
    df = df[df['masterCategory'].isin(class_counts[class_counts >= 2].index)]

X = df.drop('masterCategory', axis=1)
y = df['masterCategory']

cat_cols = X.select_dtypes(include='object').columns.tolist()
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ('num', StandardScaler(), num_cols)
])

# Logistic Regression pipeline
lr_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Random Forest pipeline
rf_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining Logistic Regression...")
lr_pipeline.fit(X_train, y_train)
y_pred_lr = lr_pipeline.predict(X_test)
acc_lr = accuracy_score(y_test, y_pred_lr)

# Train Random Forest
print("\nTraining Random Forest...")
rf_pipeline.fit(X_train, y_train)
y_pred_rf = rf_pipeline.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)

print(f"\nLogistic Regression Accuracy: {round(acc_lr * 100, 2)}%")
print(classification_report(y_test, y_pred_lr))

print(f"\nRandom Forest Accuracy: {round(acc_rf * 100, 2)}%")
print(classification_report(y_test, y_pred_rf))

# Save best model
if acc_rf > acc_lr:
    joblib.dump(rf_pipeline, "fashion_model.pkl")
    print("\nSaved: Random Forest as best model")
else:
    joblib.dump(lr_pipeline, "fashion_model.pkl")
    print("\nSaved: Logistic Regression as best model")
