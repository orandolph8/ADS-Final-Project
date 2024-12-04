import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from product_bundling import generate_bundles
from product_pricing import predict_pricing
# from territorial_sales import optimize_territories
# from lead_scoring import score_leads

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
    
    # lead_scores = score_leads(uploaded_file)
    # st.write(lead_scores)
    st.write("Lead scoring logic will be displayed here")

# Run the Streamlit app
# In terminal execute "streamlit run app.py"
if __name__ == "__main__":
    main()
