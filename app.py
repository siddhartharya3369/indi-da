
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

st.set_page_config(page_title="FitTrack AI", layout="wide")

st.title("🏋️ FitTrack AI - Professional Dashboard")

uploaded_file = st.file_uploader("Upload your dataset", type=["csv","xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith("xlsx") else pd.read_csv(uploaded_file)

    st.subheader("📊 Data Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(df))
    col2.metric("Columns", len(df.columns))
    col3.metric("Missing Values", df.isnull().sum().sum())

    st.dataframe(df.head())

    st.subheader("📈 Descriptive Analysis")

    # Categorical charts
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        fig = px.pie(df, names=col, title=f"{col} Distribution")
        st.plotly_chart(fig, use_container_width=True)

    # Numeric charts
    num_cols = df.select_dtypes(include=np.number).columns
    for col in num_cols:
        fig = px.histogram(df, x=col, title=f"{col} Distribution")
        st.plotly_chart(fig, use_container_width=True)

    # MODEL
    st.subheader("🤖 Predictive Analysis")

    df_model = df.copy()
    for col in df_model.select_dtypes(include='object').columns:
        df_model[col] = df_model[col].astype('category').cat.codes

    target = df_model.columns[-1]
    X = df_model.drop(target, axis=1)
    y = df_model[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    st.subheader("📊 Model Performance")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", round(accuracy_score(y_test, y_pred),2))
    col2.metric("Precision", round(precision_score(y_test, y_pred, average='weighted'),2))
    col3.metric("Recall", round(recall_score(y_test, y_pred, average='weighted'),2))
    col4.metric("F1 Score", round(f1_score(y_test, y_pred, average='weighted'),2))

    if len(np.unique(y)) == 2:
        y_prob = model.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig = px.line(x=fpr, y=tpr, title="ROC Curve")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Feature Importance")
    importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    fig = px.bar(importance, title="Feature Importance")
    st.plotly_chart(fig, use_container_width=True)

    # Prediction input
    st.subheader("🔮 Predict New Customer")

    input_data = {}
    for col in X.columns:
        input_data[col] = st.number_input(f"{col}", value=float(X[col].mean()))

    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        st.success(f"Prediction: {prediction}")
