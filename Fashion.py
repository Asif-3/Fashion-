import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

st.title("FASHION PREDICTION")

'''uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])'''

if uploaded_file is not None:
    df = pd.read_csv(fashion_dataset.csv)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Step 1: Select Target Variable")
    target_column = st.selectbox("Choose the target column (what you want to predict)", df.columns)

    df = df.dropna(subset=[target_column])

    df.ffill(inplace=True)
    class_counts = df[target_column].value_counts()
    if any(class_counts < 2):
        df = df[df[target_column].isin(class_counts[class_counts >= 2].index)]

    X = df.drop(columns=[target_column])
    y = df[target_column]

    
    cat_cols = X.select_dtypes(include='object').columns.tolist()
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    if not cat_cols and not num_cols:
        st.error("No valid features for training found.")
    else:
        preprocessor = ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
            ('num', StandardScaler(), num_cols)
        ])


        lr_pipeline = Pipeline([
            ('preprocess', preprocessor),
            ('classifier', LogisticRegression(max_iter=1000))
        ])

        rf_pipeline = Pipeline([
            ('preprocess', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        with st.spinner("Training Logistic Regression..."):
            lr_pipeline.fit(X_train, y_train)
            y_pred_lr = lr_pipeline.predict(X_test)
            acc_lr = accuracy_score(y_test, y_pred_lr)

        with st.spinner("Training Random Forest..."):
            rf_pipeline.fit(X_train, y_train)
            y_pred_rf = rf_pipeline.predict(X_test)
            acc_rf = accuracy_score(y_test, y_pred_rf)

 
        st.subheader("Model Performance")
        st.write(f"**Logistic Regression Accuracy:** {round(acc_lr * 100, 2)}%")
        st.text("Classification Report:\n" + classification_report(y_test, y_pred_lr))

        st.write(f"**Random Forest Accuracy:** {round(acc_rf * 100, 2)}%")
        st.text("Classification Report:\n" + classification_report(y_test, y_pred_rf))


        if acc_rf > acc_lr:
            joblib.dump(rf_pipeline, "best_model.pkl")
            st.success("Best Model Saved: Random Forest")
        else:
            joblib.dump(lr_pipeline, "best_model.pkl")
            st.success("Best Model Saved: Logistic Regression")