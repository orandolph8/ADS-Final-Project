import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import joblib

def score_leads(input_data):
    """
    Scores leads based on provided data. Trains a model if labels are available.
    """
    try:
        # Input handling: Check if input is a file or a DataFrame
        if hasattr(input_data, 'read'):  # File-like object
            df = pd.read_csv(input_data)
        elif isinstance(input_data, pd.DataFrame):  # Manual entry input
            df = input_data
        else:
            raise ValueError("Invalid input type. Expected file-like object or DataFrame.")
    except Exception as e:
        raise ValueError(f"Error reading input data: {e}")
    
    # Validate required columns
    required_features = ['business_unit', 'lead_contact', 'job_level', 
                         'industry', 'team', 'channel', 'lead_product']
    missing_features = [col for col in required_features if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Drop unnecessary columns (if present)
    df.drop(columns=['title'], errors='ignore', inplace=True)
    
    # Split data into features (X) and labels (y) if 'sales_qualified' exists
    if 'sales_qualified' in df.columns:
        X = df.drop(columns=['sales_qualified'])
        y = df['sales_qualified']
    else:  # For inference (manual scoring), no target variable exists
        X = df

    # Categorical features for preprocessing
    categorical_features = required_features
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[('cat', categorical_transformer, categorical_features)],
        remainder='passthrough'
    )
    
    # Define models for training
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'DecisionTree': DecisionTreeClassifier(random_state=42)
    }
    
    # If 'sales_qualified' exists, train a new model
    if 'sales_qualified' in df.columns:
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train and evaluate models to select the best one
        best_model = None
        best_model_name = None
        best_score = 0
        for name, model in models.items():
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])
            
            # Train the model
            pipeline.fit(X_train, y_train)
            
            # Evaluate model using AUC-ROC
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
            print(f"Model: {name}, AUC: {auc:.4f}")
            
            # Update the best model if necessary
            if auc > best_score:
                best_model = pipeline
                best_model_name = name
                best_score = auc
        
        # Save the best model for future inference
        joblib.dump(best_model, 'best_model.pkl')
    else:
        # Load the pre-trained model for inference
        try:
            best_model = joblib.load('best_model.pkl')
            best_model_name = "Pre-trained Model"
            best_score = None  # No AUC for inference-only cases
        except FileNotFoundError:
            raise ValueError("No pre-trained model found. Train a model first.")
    
    # Predict probabilities for scoring leads
    probabilities = best_model.predict_proba(X)
    
    # Define classification rules
    def classify_lead(prob):
        if prob >= 0.5:
            return 'High'
        elif 0.2 <= prob < 0.5:
            return 'Medium'
        else:
            return 'Low'
    
    # Apply classification rules
    X['probability_sales_qualified'] = [prob[1].round(2) for prob in probabilities]
    X['predicted_quality'] = X['probability_sales_qualified'].apply(classify_lead)
    
    # Return the result DataFrame with predictions and additional metadata
    return X, best_model_name, best_score if 'sales_qualified' in df.columns else None
