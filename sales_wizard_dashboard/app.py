import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from product_bundling import generate_bundles
from product_pricing import predict_pricing
# from territorial_sales import optimize_territories
from lead_scoring import score_leads

def main():
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
    
    # territorial_sales_results = optimize_territories(uploaded_file)
    # st.write(territorial_sales_results)
    st.write("Territorial sales optimization logic will be displayed here")

def lead_scoring_module(uploaded_file):
    st.header("Lead Scoring Module")
    
    # Perform Lead Scoring
    lead_scoring = score_leads(uploaded_file)
    
    # Tabs for Input Options
    tab1, tab2 = st.tabs(["Upload CSV File", "Enter Lead Details Manually"])

    # Variable to Store Results
    result_df, best_model_name, best_model_auc = None, None, None

    # Tab 1: Upload a CSV File
    with tab1:
        st.subheader("Upload a CSV File")
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
        if uploaded_file:
            result_df, best_model_name, best_model_auc = score_leads(uploaded_file)
            if result_df is not None:
                st.success("File processed successfully")
            else:
                st.error("Error processing the file.")

    # Tab 2: Manual Lead Entry
    with tab2:
        st.subheader("Enter Lead Details Manually")
        
        # Dropdown input fields with predefined options
        business_unit = st.selectbox("Business Unit", [
            'business_unit_1', 'business_unit_2', 'business_unit_3', 'business_unit_4'
        ])
        lead_contact = st.selectbox("Lead Contact", ['Contact', 'Lead'])
        job_level = st.selectbox("Job Level", [
            'Staff-Level', 'Unknown', 'Director-Level', 'Manager-Level', 'C-Level',
            'Provider-Level', 'VP-Level'
        ])
        industry = st.selectbox("Industry", [
            'Assisted Living Facility', 'Home Health', 'Ambulatory Health Care Facilities', 
            'Physician', 'Health System', 'Skilled Nursing Facility', 'Dental', 
            'Software Vendor', 'Hospital', 'Other Ambulatory Provider', 'PT/OT/Rehab', 
            'DME & Medical Supplies', 'Other', 'Health Plan', 'Hospice', 'Mental Health', 
            'Revenue Cycle Management', 'Chiropractic', 'Billing Services', 'Unknown', 
            'Pharmaceutical', 'Health Plan / Payer', 'Eye and Vision Services', 
            'Facility/Agency', 'Residential Treatment Facilities', 
            'Speech, Language and Hearing Service', 'Respiratory, Rehab & Restorative Service', 
            'Pharmacy Service', 'Transportation Services', 'Podiatry', 'Nursing Service', 
            'Consulting Services', 'Clinical Research', 'Management Companies', 'Pharmacy', 
            'Behavioral Health & Social Service', 'Physician / Physician Group', 
            'Financial Services', 'Patient/Disease/Advocacy Group', 'Professional', 
            'Laboratories', 'Vendor', 'Biotech', 'Emergency Medical Service', 'Long Term Care', 
            'ACO', 'Insight', 'ACOs-MCOs-IDNs', 'Staffing Agency', 'Payer Services', 
            'Medical Device Manufacturer', 'Professional Society', 'Data Licensing Partner', 
            'Other Service', 'Health Plan/Payer', 'Dietary & Nutritional Service', 
            'Biotech Manufacturer', 'Pharma Manufacturer', 'Therapeutics', 'Sotfware Vendor', 
            'Non-Healthcare', 'Trade Association', 'Diagnostics', 'Need Review', 
            'Correctional Facility'
        ])
        team = st.selectbox("Team", [
            'team_1', 'team_2', 'team_3', 'team_4', 'team_5', 'team_6', 'team_7', 'team_8',
            'team_9', 'team_10', 'team_11', 'team_12', 'team_13'
        ])
        channel = st.selectbox("Channel", [
            'Webinar', 'MDR Meeting', 'Inbound Call', 'Demo Request', 'PPC',
            'Content Syndication', 'Survey', 'Web Chat', 'Tradeshow', 'PBP', 'Case Study',
            'Web Traffic', 'Unknown'
        ])
        lead_product = st.selectbox("Lead Product", [
            'product_7', 'product_27', 'product_40', 'product_1', 'product_41', 'product_28', 
            'product_37', 'product_49', 'product_4', 'product_57', 'product_47', 'product_3', 
            'product_24', 'product_5', 'product_35', 'product_46', 'product_19', 'product_23', 
            'product_14', 'product_50', 'product_13', 'product_34', 'product_22', 'product_64', 
            'product_6', 'product_53', 'product_26', 'product_44', 'product_25', 'product_42', 
            'product_48', 'product_20', 'product_16', 'product_32', 'product_56', 'product_12', 
            'product_60', 'product_51', 'product_61', 'product_8', 'product_55', 'product_21', 
            'product_38', 'product_67', 'product_62', 'product_15', 'Unknown', 'product_29', 
            'product_11', 'product_39', 'product_52', 'product_33', 'product_68', 'product_43', 
            'product_65', 'product_18', 'product_9', 'product_63', 'product_59', 'product_66', 
            'product_31', 'product_69', 'product_45', 'product_36', 'product_30', 'product_54', 
            'product_10', 'product_58', 'product_17', 'product_2'
        ])
        numerical_feature = st.number_input("Numerical Feature (e.g., Sales Volume)", value=0.0)
        
        # Collect user inputs into a dictionary
        user_inputs = {
            "business_unit": business_unit,
            "lead_contact": lead_contact,
            "job_level": job_level,
            "industry": industry,
            "team": team,
            "channel": channel,
            "lead_product": lead_product,
            "numerical_feature": numerical_feature
        }
        
        # Button for prediction
        if st.button("Score Lead"):
            try:
                result_df, best_model_name, best_model_auc = score_leads(pd.DataFrame([user_inputs]))
                if result_df is not None:
                    st.success("Lead scored successfully!")
                else:
                    st.error("Error scoring the lead.")
            except Exception as e:
                st.error(f"Error: {e}")

    # Display Results
    if result_df is not None:
        st.subheader("Prediction Results")
        st.write(f"**Best Model**: {best_model_name}")
        st.write(f"**AUC Score**: {best_model_auc:.4f}")
        st.dataframe(result_df)
        
        # Option to download results as CSV
        csv = result_df.to_csv(index=False)
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="lead_scoring_results.csv",
            mime="text/csv"
        )

# Run the Streamlit app
# In terminal execute "streamlit run app.py"
if __name__ == "__main__":
    main()
