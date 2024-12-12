import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from joblib import dump, load
import os

def predict_pricing(file, model_path="models/saved_pipeline.joblib"):
    try:
        # Check if a pre-trained pipeline exists
        if os.path.exists(model_path):
            print(f"Loading pre-trained pipeline from {model_path}")
            pipeline = load(model_path)
        else:
            # Read the input CSV file
            df = pd.read_csv(file)

            # Features and target variable
            X = df[["product_name", "product_category", "region", "customer_segment", "quantity_sold"]]
            y = df["price_per_unit"]

            categorical_features = ["product_name", "product_category", "region", "customer_segment"]
            numerical_features = ["quantity_sold"]

            # Preprocessing pipeline
            preprocessor = ColumnTransformer(transformers=[
                ("num", StandardScaler(), numerical_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
            ])

            # Define models
            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                "SVR": SVR()
            }

            # Split data into training and testing
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Evaluate models and store results
            results = {}
            for name, model in models.items():
                pipeline = Pipeline(steps=[
                    ("preprocessor", preprocessor),
                    ("regressor", model)
                ])
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                results[name] = rmse

            # Identify the best model
            best_model_name = min(results, key=results.get)
            best_model_rmse = results[best_model_name]

            # Train best model on the full dataset
            best_model = models[best_model_name]
            pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("regressor", best_model)
            ])
            pipeline.fit(X, y)

            # Save the trained pipeline
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            dump(pipeline, model_path)
            print(f"Pipeline saved at {model_path}")

        # Generate predictions for the input file
        df = pd.read_csv(file)
        df["recommended_price"] = pipeline.predict(df[["product_name", "product_category", "region", "customer_segment", "quantity_sold"]])

        # Generate ideal pricing recommendations
        ideal_pricing = df.groupby(["region", "customer_segment", "product_name"])["recommended_price"].mean().round(2).reset_index()

        return results, best_model_name, best_model_rmse, ideal_pricing

    except Exception as e:
        raise ValueError(f"An error occurred while predicting pricing: {e}")
