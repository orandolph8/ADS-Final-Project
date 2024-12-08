#Import Libraries and Packages
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# Define Healthcare Sales Territory Optimization function
def optimize_territories(file):

  #Upload File
  df = pd.read_csv(file)

  # Extract 'state' from 'address' column
  df['state'] = df['address'].str.split(', ').str[-1]

  # Map states to regions
  state_to_region = df[['state', 'region']].drop_duplicates().set_index('state')['region'].to_dict()

  # Map and aggregate sales
  region_sales = df.groupby('region')['sale_amount'].sum()
  state_sales = [{'state': state, 'sale_amount': region_sales[region]}
                 for state, region in state_to_region.items()]
  state_sales_df = pd.DataFrame(state_sales)

  # Convert 'date_of_sale' to datetime for time series analysis
  df['date_of_sale'] = pd.to_datetime(df['date_of_sale'])
  df['year_month'] = df['date_of_sale'].dt.to_period('M') # Extract only year and month

  # Calculate growth rate by region and time
  regional_sales = df.groupby(['state', 'year_month'])['sale_amount'].sum().reset_index()
  regional_sales['growth_rate'] = regional_sales.groupby('state')['sale_amount'].pct_change() * 100
  regional_sales['growth_rate'] = regional_sales['growth_rate'].fillna(0) # Replace NaN growth rates with 0
  
   # Merge growth_rate back into original dataframe
  df = df.merge(
    regional_sales[['state', 'growth_rate']], on='state', how='left')
  
  # Categorize growth into high and low with quantiles
  df['growth_category'] = pd.qcut(df['growth_rate'], q=2, labels=['Low Growth', 'High Growth'])

  # One-Hot Encoding for 'region' column
  region_dummies = pd.get_dummies(df['region'], prefix='region')
  df = pd.concat([df, region_dummies], axis=1)
  
  # Define Features
  features = ['quantity_sold', 'price_per_unit', 'growth_rate', 'sale_amount',
              'region_West', 'region_South', 'region_Northeast',
              'region_Pacific Northwest/Mountain', 'multiple_items']
  target = 'growth_category_encoded'

  # Apply PCA for dimensionality reduction with a 95% explained variance
  pca = PCA(n_components=0.95, random_state=42)
  df_pca = pca.fit_transform(df[features])
  df_pca = pd.DataFrame(df_pca)
  
  # Encode Target Variable
  if target not in df.columns or df[target].dtype == 'object':
    df['growth_category_encoded'] = LabelEncoder().fit_transform(df['growth_category'])

  #Split data into training and testing sets
  X = df_pca
  y = df['growth_category_encoded']
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Standardize numerical features
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)
  
  # Train Random Forest
  model = RandomForestClassifier(n_estimators=1000, max_depth=5, random_state=42)
  model.fit(X_train_scaled, y_train)

  # Predict High Growth Categories
  y_pred = model.predict(X_test_scaled)

  # Udpate only rows in df corresponding to test set
  df.loc[X_test.index, 'predicted_growth_category'] = y_pred

  # Filter High Growth predictions
  df_high_growth = df[df['predicted_growth_category'] == 1]

  # Aggregate sales amount by state
  high_growth_states = df_high_growth.groupby('state').agg({
    'growth_rate': 'mean', # Average growth rate for high-growth states
    'sale_amount': 'sum', # Total sales amount for high-growth states
    'predicted_growth_category': 'count' # Number of high-growth predictions
  }).reset_index()

  # Create 'growth_rate_percentage'
  high_growth_states['growth_rate_percentage'] = high_growth_states['growth_rate']

  # Handle missing states
  all_states = pd.DataFrame(df['state'].unique(), columns=['state'])
  high_growth_states = all_states.merge(high_growth_states, on='state', how='left').fillna({
    'growth_rate': 0,
    'growth_rate_percentage': 0,
    'predicted_growth_category': 0,
    'sale_amount': 0
  })

  # Create interactive choropleth map
  fig = px.choropleth(
      high_growth_states,
      locations='state',
      locationmode='USA-states',
      color='predicted_growth_category',
      color_continuous_scale='Greens',
      scope='usa',
      hover_name='state',
      hover_data={'growth_rate_percentage': ':.2f', 'sale_amount': ':.2f'},
      title='Predicted High Growth States with Growth Rates'
  )

  # Updating hover template for better readability
  fig.update_traces(
    hovertemplate=(
      '<b>State:</b> %{hovertext}<br>'
      '<b>Predicted High-Growth Count:</b> %{z}<br>'
      '<b>Avg Growth Rate (%):</b> %{customdata[0]:.2%}<br>'
      '<b>Total Sales Amount:</b> $%{customdata[1]:,.2f}'
    ),
    customdata=high_growth_states[['growth_rate_percentage', 'sale_amount']]
  )

  return fig
