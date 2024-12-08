#Import Libraries and Packages
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Define Healthcare Sales Territory Optimization function
def optimize_territories(file):

  #Upload File
  df = pd.read_csv(file)

  # Convert 'date_of_sale' to datetime
  df['date_of_sale'] = pd.to_datetime(df['date_of_sale'])

  # Group by region and year_month
  df['year_month'] = df['date_of_sale'].dt.to_period('M') # Extract only year and month
  regional_sales = df.groupby(['region', 'year_month'])['sale_amount'].sum().reset_index()

  # Calculate percentage change for growth_rate
  regional_sales['growth_rate'] = regional_sales.groupby('region')['sale_amount'].pct_change() * 100

  # Merge growth_rate back into original dataframe
  df = df.merge(regional_sales[['region', 'year_month', 'growth_rate']], on=['region', 'year_month'], how='left')

  # Categorize growth into high and low with quantiles
  df['growth_category'] = pd.qcut(df['growth_rate'], q=2, labels=['Low Growth', 'High Growth'])

  # One-Hot Encoding for 'region' column
  region_dummies = pd.get_dummies(df['region'], prefix='region')
  df = pd.concat([df, region_dummies], axis=1)
  
  # Define Features and Target
  features = ['quantity_sold', 'price_per_unit', 'growth_rate', 'sale_amount',
              'region_West', 'region_South', 'region_Northeast',
              'region_Pacific Northwest/Mountain', 'multiple_items']
  target = 'growth_category_encoded'

  # Encode Target Variable
  if target not in df.columns or df[target].dtype == 'object':
    df['growth_category_encoded'] = LabelEncoder().fit_transform(df['growth_category'])


  # Standardize numerical features
  scaler = StandardScaler()
  df[features] = scaler.fit_transform(df[features])

  #Split data into training and testing sets
  X = df[features]
  y = df[target]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Train Random Forest
  model = RandomForestClassifier(n_estimators=1000, max_depth=5, random_state=42)
  model.fit(X_train, y_train)

  # Predict High Growth Categories
  df['predicted_growth_category'] = model.predict(X)

  # Filter High Growth predictions
  high_growth_states = df[df['predicted_growth_category'] == 1].groupby('state')['sale_amount'].sum().reset_index()

  # Create interactive choropleth map
  fig = px.choropleth(
      high_growth_states,
      locations='state',
      locationmode='USA-states',
      color='sale_amount',
      color_continuous_scale='Viridis',
      scope='usa',
      hover_name='state',
      hover_data=['sale_amount'],
      title='Predicted High Growth Sales by State'
  )

  return fig
