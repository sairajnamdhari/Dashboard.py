import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

# App Config
st.set_page_config(page_title="Agent AI Trainer", page_icon="🤖", layout="centered")
st.title("🤖 Agent AI: Train & Predict App")

st.markdown("""
Welcome to **Agent AI Trainer**!  
Upload your dataset, select the target column, and choose from multiple ML models to train and make predictions.
""")

# Upload training data
train_file = st.file_uploader("📂 Upload your **training dataset (CSV)**", type=["csv"])

if train_file:
    df = pd.read_csv(train_file)
    st.markdown("### 👀 Preview of your training data")
    st.dataframe(df.head())

    # Target column
    target_col = st.selectbox("🎯 Select the target column (what to predict)", df.columns)

    if target_col:
        df = df.dropna()
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Label Encode categorical features in X
        label_encoders = {}
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                label_encoders[col] = le

        # Encode target column if it's categorical
        if y.dtype == 'object':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y)
        else:
            le_target = None

        # Auto detect classification or regression
        task_type = 'classification' if len(set(y)) < 20 else 'regression'
        st.info(f"🧠 Detected Task Type: **{task_type.capitalize()}**")

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model Selection
        st.markdown("### 🤖 Select Model for Training")
        if task_type == 'classification':
            model_name = st.selectbox("Choose a classification model", [
                "Random Forest", "Logistic Regression", "Support Vector Machine",
                "K-Nearest Neighbors", "XGBoost"])
            models_dict = {
                "Random Forest": RandomForestClassifier(),
                "Logistic Regression": LogisticRegression(),
                "Support Vector Machine": SVC(),
                "K-Nearest Neighbors": KNeighborsClassifier(),
                "XGBoost": XGBClassifier()
            }
        else:
            model_name = st.selectbox("Choose a regression model", [
                "Random Forest", "Linear Regression", "Support Vector Regressor",
                "K-Nearest Neighbors", "XGBoost"])
            models_dict = {
                "Random Forest": RandomForestRegressor(),
                "Linear Regression": LinearRegression(),
                "Support Vector Regressor": SVR(),
                "K-Nearest Neighbors": KNeighborsRegressor(),
                "XGBoost": XGBRegressor()
            }

        model = models_dict[model_name]

        # Train selected model
        st.markdown(f"### 🛠️ Training `{model_name}` Model")
        with st.spinner("Training..."):
            model.fit(X_train, y_train)
        st.success("✅ Model training complete!")

        # User Input for Prediction
        st.markdown("## 🔮 Make Predictions on New Data")
        st.markdown("Enter new data for prediction:")

        user_input = {}
        for col in X.columns:
            example_value = df[col].iloc[0]
            value = st.text_input(f"{col}", value=str(example_value))

            # Try to convert input if it was encoded
            if col in label_encoders:
                le = label_encoders[col]
                try:
                    value = le.transform([value])[0]
                except:
                    st.error(f"⚠️ Invalid value for column `{col}`. Please enter one of: {list(le.classes_)}")
                    st.stop()
            else:
                try:
                    value = float(value)
                except:
                    st.error(f"⚠️ `{col}` must be a number.")
                    st.stop()

            user_input[col] = value

        if st.button("🚀 Predict"):
            input_df = pd.DataFrame([user_input])
            prediction = model.predict(input_df)[0]

            # Decode prediction if target was label encoded
            if le_target:
                prediction = le_target.inverse_transform([prediction])[0]

            st.success(f"📈 Prediction Result: `{prediction}`")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit | Powered by Scikit-Learn & XGBoost")
