#Import Libraries and Packages
from sklearn.cluster import KMeans
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
#from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

# Define Healthcare Sales Territory Optimization function
def optimize_territories(file):

  #Upload File
  df = pd.read_csv(file)

  # Extract 'state' from 'address' column
  df['state'] = df['address'].str.split(', ').str[-1]

  # Ensure 'state' column is clean
  df['state'] = df['state'].str.strip()

  # Convert 'date_of_sale' to datetime for time series analysis
  df['date_of_sale'] = pd.to_datetime(df['date_of_sale'])
  df['year_month'] = df['date_of_sale'].dt.to_period('M') # Extract only year and month

  # Calculate growth rate by state and time
  state_sales = df.groupby(['state', 'year_month'])['sale_amount'].sum().reset_index()
  state_sales['growth_rate'] = state_sales.groupby('state')['sale_amount'].pct_change()
  state_sales['growth_rate'] = state_sales['growth_rate'].fillna(0) # Replace NaN growth rates with 0
  
  # Merge growth_rate back into original dataframe
  df = df.merge(
    state_sales[['state', 'growth_rate']], on='state', how='left')
  
  # Apply K-Means clustering to categorize into 'High Growth' and 'Low Growth'
  kmeans = KMeans(n_clusters=2, random_state=42)
  df['growth_category'] = kmeans.fit_predict(df[['growth_rate']])

  # Map cluster labels to meaningful categories (0=Low Growth, 1=High Growth)
  df['growth_category'] = df['growth_category'].apply(lambda x: 'High Growth' if x == 1 else 'Low Growth')
  
  # One-Hot Encoding for 'state' column
  state_dummies = pd.get_dummies(df['state'], prefix='state')
  df = pd.concat([df, state_dummies], axis=1)
  
  # Define Features
  features = ['quantity_sold', 'price_per_unit', 'growth_rate', 'sale_amount',
              'multiple_items'] + list(state_dummies.columns)
  target = 'growth_category_encoded'

  # Apply PCA for dimensionality reduction with 4 components
  pca = PCA(n_components=4, random_state=42)
  df_pca = pca.fit_transform(df[features])
  df_pca = pd.DataFrame(df_pca, columns=[f'PC{i+1}' for i in range(4)])
  
  # Encode Target Variable
  if target not in df.columns or df[target].dtype == 'object':
    df['growth_category_encoded'] = LabelEncoder().fit_transform(df['growth_category'])

  # Define X and y
  X = df_pca
  y = df['growth_category_encoded']

  # Split into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  
  # Standardize numerical features
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)
  
  # Train XGBoost Classifier
  model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=None,
    random_state=42,
    n_jobs=-1
  )

  #Train model
  model.fit(X_train_scaled, y_train)

  # Predict with best model
  y_pred = model.predict(X_test_scaled)

  # Calculate Accuracy
  accuracy = accuracy_score(y_test, y_pred)

  # Udpate only rows in df corresponding to test set
  df.loc[X_test.index, 'predicted_growth_category'] = y_pred

  # Calculate average growth rate percentage by state
  average_growth_by_state = df.groupby('state')['growth_rate'].mean()

  # Calculate total sales by state
  total_sales_by_state = df.groupby('state')['sale_amount'].sum()

  # Merge into DataFrame
  all_growth_states = pd.DataFrame({
    'growth_rate': average_growth_by_state,
    'sale_amount': total_sales_by_state
  }).reset_index()

  # Aggregate count of growth predictions
  growth_counts = df.groupby('state')['predicted_growth_category'].count().reset_index()
  growth_counts = growth_counts.rename(columns={'predicted_growth_category': 'predicted_growth_category_count'})

  # Merge into main DataFrame
  all_growth_states = pd.merge(all_growth_states, growth_counts, on='state', how='left')

  # Handle missing states
  all_states = pd.DataFrame(df['state'].unique(), columns=['state'])
  all_growth_states = all_states.merge(all_growth_states, on='state', how='left').fillna({
    'growth_rate': 0,
    'predicted_growth_category': 0,
    'sale_amount': 0
  })

  # Create interactive choropleth map
  fig = px.choropleth(
      all_growth_states,
      locations='state',
      locationmode='USA-states',
      color='predicted_growth_category_count',
      color_continuous_scale='Greens',
      scope='usa',
      hover_name='state',
      hover_data={'growth_rate': ':.2f', 'sale_amount': ':.2f'},
      title='Predicted Growth States'
  )

  # Updating hover template for better readability
  fig.update_traces(
    hovertemplate=(
      '<b>State:</b> %{hovertext}<br>'
      '<b>Growth Count:</b> %{z}<br>'
      '<b>Avg Growth Rate (%):</b> %{customdata[0]:.2%}<br>'
      '<b>Total Sales Amount:</b> $%{customdata[1]:,.2f}'
    ),
    customdata=all_growth_states[['growth_rate', 'sale_amount']]
  )

  return fig, accuracy
