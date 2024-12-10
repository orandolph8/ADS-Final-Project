import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from PIL import Image

from product_bundling import generate_bundles
from product_pricing import predict_pricing
from territorial_sales import optimize_territories
from lead_scoring import score_leads

# Add background image for main    
def add_bg_from_local(image_file):
    with open(image_file, 'rb') as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp{{
        background-image: url(data:image/{'jpeg'};base64,{encoded_string.decode()});
        background-size: cover;
        background-position: center;
    }}
    <style/>
    """,
    unsafe_allow_html=True
    )

# Add background to sidebar
def set_sidebar_background(image_file):
    with open(image_file, 'rb') as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    [data-testid="stSidebar"] {{
        background-image: url(data:image/{'png'};base64,{encoded_string.decode()});
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    <style/>
    """,
    unsafe_allow_html=True
    )

# Set font color
def set_font_color(main_color: str):
    st.markdown(
        f"""
        <style>
        /* Change font color for the main app content */
        .stApp {{
            color: {main_color};
        }}
        /* Exclude sidebar from the font color change */
        section[data-testid="stSidebar"] * {{
            color: inherit !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def main():
    # Condense padding
    padding = 0
    st.markdown(f""" <style>
        .reportview-container .main .block-container{{
            padding-top: {padding}rem;
            padding-right: {padding}rem;
            padding-left: {padding}rem;
            padding-bottom: {padding}rem;
        }} </style> """, unsafe_allow_html=True)
    
    # Set font color for main app
    set_font_color('#FFFFFF')
    
    # Add background image
    add_bg_from_local('healthcare_products_right.jpeg')

    # Add sidebar background
    set_sidebar_background('medical_products_sidebar_2dicaldical.png')
    
    st.title("Healthcare SalesWizard Dashboard")

    # Radio Buttons on the side for Module Selection
    st.sidebar.header("Module Navigation")
    selected_section = st.sidebar.radio(
        "Go to",
        ("Product Pricing", "Product Bundling", "Territorial Sales Optimization", "Lead Scoring")
    )

    # Shared file uploader that works for all modules
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

    # Only run the relevant module based on user selection
    if uploaded_file is not None:
        if selected_section == "Product Pricing":
            product_pricing_module(uploaded_file)

        elif selected_section == "Product Bundling":
            product_bundling_module(uploaded_file)

        elif selected_section == "Territorial Sales Optimization":
            territorial_sales_module(uploaded_file)

        elif selected_section == "Lead Scoring":
            lead_scoring_module(uploaded_file)

def product_pricing_module(uploaded_file):
    # Perform analysis
    results, best_model_name, best_model_rmse, ideal_pricing = predict_pricing(uploaded_file)

    # Display results
    st.subheader("Model Performance")
    st.write(pd.DataFrame.from_dict(results, orient="index", columns=["RMSE"]).sort_values(by="RMSE"))

    st.subheader("Best Model")
    st.write(f"The best model is **{best_model_name}** with an RMSE of **{best_model_rmse:.2f}**.")

    st.subheader("Pricing Recommendations")
    st.dataframe(ideal_pricing)

    # Visualization 1: Recommended Price Distribution
    st.subheader("Visualization: Recommended Price Distribution")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.histplot(ideal_pricing["recommended_price"], kde=True, ax=ax1, bins=20, color="blue")
    ax1.set_title("Distribution of Recommended Prices")
    ax1.set_xlabel("Recommended Price")
    ax1.set_ylabel("Frequency")
    st.pyplot(fig1)

    # Visualization 2: Region Average Pricing
    st.subheader("Visualization: Region-wise Average Recommended Price")
    region_avg = ideal_pricing.groupby("region")["recommended_price"].mean().sort_values()
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    region_avg.plot(kind="bar", ax=ax2, color="orange", alpha=0.8)
    ax2.set_title("Average Recommended Price by Region")
    ax2.set_ylabel("Average Recommended Price")
    ax2.set_xlabel("Region")
    st.pyplot(fig2)

def product_bundling_module(uploaded_file):
    # Perform bundle analysis
    hospital_avg, clinic_avg = generate_bundles(uploaded_file)

    # Display results for Hospital
    st.subheader("Hospital Bundle Recommendations")
    st.write("Recommended Quantities for the Hospital Bundle:")
    st.write(hospital_avg.round(0))

    # Visualization for Hospital
    st.subheader("Visualization: Hospital Bundle Quantities")
    fig, ax = plt.subplots(figsize=(10, 6))
    hospital_avg.sort_values().plot(kind="bar", ax=ax, color="skyblue")
    ax.set_title("Recommended Quantities for Hospital Bundle")
    ax.set_ylabel("Quantity Sold")
    ax.set_xlabel("Product Name")
    st.pyplot(fig)

    # Display results for Clinic
    st.subheader("Clinic Bundle Recommendations")
    st.write("Recommended Quantities for the Clinic Bundle:")
    st.write(clinic_avg.round(0))

    # Visualization for Clinic
    st.subheader("Visualization: Clinic Bundle Quantities")
    fig, ax = plt.subplots(figsize=(10, 6))
    clinic_avg.sort_values().plot(kind="bar", ax=ax, color="lightgreen")
    ax.set_title("Recommended Quantities for Clinic Bundle")
    ax.set_ylabel("Quantity Sold")
    ax.set_xlabel("Product Name")
    st.pyplot(fig)

def territorial_sales_module(uploaded_file):
    st.header("Territorial Sales Optimization Module")
    st.subheader('Please be patient while model optimizes')

    try:
        # Generate choropleth map    
        territorial_sales_results, best_params, accuracy = optimize_territories(uploaded_file)

        # Display map
        st.subheader('Predicted Growth Sales by State')
        st.plotly_chart(territorial_sales_results)

        # Display best hyperparameters
        st.subheader('Best Hyperparameters')
        st.write(best_params)

        # Display accuracy
        st.subheader('Model Accuracy')
        st.write(f'Accuracy: {accuracy * 100:.2f}%')

    except Exception as e:
        
        st.error(f'An error occurred while processing the file: {e}')

def lead_scoring_module(uploaded_file):
    st.header("Lead Scoring Module")

    # Initialize session state for results
    if "result_df" not in st.session_state:
        st.session_state.result_df = None
    if "manual_entry_result" not in st.session_state:
        st.session_state.manual_entry_result = None

    # Tabs for Input Options
    tab1, tab2 = st.tabs(["Upload CSV File", "Enter Lead Details Manually"])

    # Tab 1: Upload a CSV File
    with tab1:
        st.subheader("Upload a CSV File")

        if uploaded_file:
            try:
                # Process and score the uploaded CSV file
                st.session_state.result_df, st.session_state.best_model_name, st.session_state.best_model_auc = score_leads(uploaded_file)
                st.success("File processed successfully!")
            except Exception as e:
                st.error(f"Error processing the file: {e}")

        # Display results for the uploaded CSV file
        if st.session_state.result_df is not None:
            st.subheader("Prediction Results (CSV File)")

            # Extract and display the first lead's probability and quality
            first_probability = st.session_state.result_df.iloc[0]['probability_sales_qualified']
            first_quality = st.session_state.result_df.iloc[0]['predicted_quality']
            st.write(f"The lead has a probability of **{first_probability:.2f}** to be sales qualified, "
                     f"which corresponds to a **{first_quality}** rating.")

            # Display the full results DataFrame
            st.dataframe(st.session_state.result_df)

            # Include a note explaining classification rules
            st.write("""
                **Classification Rules:**
                - High: Probability >= 0.5  
                - Medium: Probability between 0.2 and 0.5  
                - Low: Probability < 0.2
            """)

            # Option to download the results as a CSV
            csv = st.session_state.result_df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="lead_scoring_results.csv",
                mime="text/csv"
            )

    # Tab 2: Manual Lead Entry
    with tab2:
        st.subheader("Enter Lead Details Manually")

        # Dropdown input fields for manual data entry
        business_unit = st.selectbox("Business Unit", ['business_unit_1', 'business_unit_2', 'business_unit_3', 'business_unit_4'])
        lead_contact = st.selectbox("Lead Contact", ['Contact', 'Lead'])
        job_level = st.selectbox("Job Level", ['Staff-Level', 'Unknown', 'Director-Level', 'Manager-Level', 'C-Level', 'Provider-Level', 'VP-Level'])
        industry = st.selectbox("Industry", ['Assisted Living Facility', 'Home Health', 'Ambulatory Health Care Facilities', 'Physician', 'Health System', 'Skilled Nursing Facility', 'Dental', 'Software Vendor', 'Hospital'])
        team = st.selectbox("Team", ['team_1', 'team_2', 'team_3', 'team_4', 'team_5'])
        channel = st.selectbox("Channel", ['Webinar', 'MDR Meeting', 'Inbound Call', 'Demo Request', 'PPC'])
        lead_product = st.selectbox("Lead Product", ['product_1', 'product_2', 'product_3', 'product_4', 'product_5'])

        # Collect user inputs into a dictionary
        user_inputs = {
            "business_unit": business_unit,
            "lead_contact": lead_contact,
            "job_level": job_level,
            "industry": industry,
            "team": team,
            "channel": channel,
            "lead_product": lead_product,
        }

        # Button to score the manually entered lead
        if st.button("Score Lead"):
            try:
                # Perform scoring for manually entered lead data
                input_df = pd.DataFrame([user_inputs])  # Convert user inputs to DataFrame
                st.session_state.manual_entry_result, _, _ = score_leads(input_df)
                st.success("Lead scored successfully!")
            except Exception as e:
                st.error(f"Error: {e}")

        # Display results for the manually entered lead
        if st.session_state.manual_entry_result is not None:
            st.subheader("Prediction Results (Manual Entry)")

            # Extract probability and predicted quality for the first lead
            first_probability = st.session_state.manual_entry_result.iloc[0]['probability_sales_qualified']
            first_quality = st.session_state.manual_entry_result.iloc[0]['predicted_quality']
            st.write(f"The lead has a probability of **{first_probability:.2f}** to be sales qualified, "
                     f"which corresponds to a **{first_quality}** rating.")

            # Display the full results DataFrame
            st.dataframe(st.session_state.manual_entry_result)

            # Include a note explaining classification rules
            st.write("""
                **Classification Rules:**
                - High: Probability >= 0.5  
                - Medium: Probability between 0.2 and 0.5  
                - Low: Probability < 0.2
            """)

            # Option to download the results as a CSV
            csv = st.session_state.manual_entry_result.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="manual_lead_scoring_results.csv",
                mime="text/csv"
            )


# Run the Streamlit app
# In terminal execute "streamlit run app.py"
if __name__ == "__main__":
    main()
