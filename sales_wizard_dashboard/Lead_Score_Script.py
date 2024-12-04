import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

def score_leads(file):
        # Load dataset from the provided file
        df = pd.read_csv(file)
        
        # Drop unnecessary columns
        df.drop(columns=['title'], errors='ignore', inplace=True)
        
        # Define features and target
        X = df.drop(columns=['sales_qualified'])
        y = df['sales_qualified']
        
        # Categorical features for preprocessing
        categorical_features = ['business_unit', 'lead_contact', 'job_level', 
                                'industry', 'team', 'channel', 'lead_product']
        
        # Preprocessing: One-hot encoding for categorical features
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        preprocessor = ColumnTransformer(
            transformers=[('cat', categorical_transformer, categorical_features)],
            remainder='passthrough'  
        )
        
        # Define models to evaluate
        models = {
            'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
            'RandomForest': RandomForestClassifier(random_state=42),
            'GradientBoosting': GradientBoostingClassifier(random_state=42),
            'DecisionTree': DecisionTreeClassifier(random_state=42)
        }
        
        # Train Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Evaluate models to select the best one
        best_model = None
        best_model_name = None
        best_score = 0
        for name, model in models.items():
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])
            
            # Fit the model
            pipeline.fit(X_train, y_train)
            
            # Evaluate model using AUC-ROC
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
            print(f"Model: {name}, AUC: {auc:.4f}")
            
            # Allow for Best Model Update
            if auc > best_score:
                best_model = pipeline
                best_model_name = name
                best_score = auc
        
        # Fit the best model on the entire dataset
        best_model.fit(X, y)
        
        # Predict probabilities for scoring leads
        probabilities = best_model.predict_proba(X_test)
        
        # Define classification rules based on probability
        def classify_lead(prob):
            if prob >= 0.5:
                return 'High'
            elif 0.2 <= prob < 0.5:
                return 'Medium'
            else:
                return 'Low'
        
        # Apply classification rules
        predictions = [classify_lead(prob[1]) for prob in probabilities]
        
        # Prepare and return result DataFrame
        result_df = X_test.copy()
        result_df['predicted_quality'] = predictions
        result_df['probability_sales_qualified'] = [prob[1].round(2) for prob in probabilities]
        
        return result_df, best_model_name, best_score