import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

def generate_bundles(file):
    df = pd.read_csv(file)

    # Filter and process data for hospital and clinic customer segments
    basket_hospital = (df[df['customer_segment'] == "Hospital"]
          .groupby(['transaction_id', 'product_name'])['quantity_sold']
          .sum().unstack().reset_index().fillna(0)
          .set_index('transaction_id'))

    basket_clinic = (df[df['customer_segment'] == "Clinic"]
          .groupby(['transaction_id', 'product_name'])['quantity_sold']
          .sum().unstack().reset_index().fillna(0)
          .set_index('transaction_id'))

    # Hot encode the data
    def hot_encode(x):
        if x <= 0:
            return 0
        if x >= 1:
            return 1

    basket_hospital = basket_hospital.applymap(hot_encode)
    basket_clinic = basket_clinic.applymap(hot_encode)

    # Generate frequent itemsets and association rules
    frequent_items_hospital = apriori(basket_hospital, min_support=.01, use_colnames=True)
    frequent_items_clinic = apriori(basket_clinic, min_support=.005, use_colnames=True)
    
    num_itemsets_hospital = len(frequent_items_hospital)
    rules_hospital = association_rules(frequent_items_hospital, num_itemsets=num_itemsets_hospital, metric="lift", min_threshold=0.5)

    num_itemsets_clinic = len(frequent_items_clinic)
    rules_clinic = association_rules(frequent_items_clinic, num_itemsets=num_itemsets_clinic, metric="lift", min_threshold=0.5)

    # Sort rules by confidence and lift
    rules_hospital = rules_hospital.sort_values(['confidence', 'lift'], ascending=[False, False])
    rules_clinic = rules_clinic.sort_values(['confidence', 'lift'], ascending=[False, False])

    # Extract top rule for Hospital
    top_rule_hospital = rules_hospital.iloc[0]
    top_antecedent_hospital = list(top_rule_hospital['antecedents'])[0]
    top_consequent_hospital = list(top_rule_hospital['consequents'])[0]
    
    hospital_transactions_df = df[
        (df['customer_segment'] == 'Hospital') & 
        (df['product_name'].isin([top_antecedent_hospital, top_consequent_hospital]))
    ]

    hospital_bundle_quantities = hospital_transactions_df.groupby(['transaction_id', 'product_name'])['quantity_sold'].sum().unstack()
    hospital_average_quantities = hospital_bundle_quantities.mean()

    # Extract top rule for Clinic
    top_rule_clinic = rules_clinic.iloc[0]
    top_antecedent_clinic = list(top_rule_clinic['antecedents'])[0]
    top_consequent_clinic = list(top_rule_clinic['consequents'])[0]
    
    clinic_transactions_df = df[
        (df['customer_segment'] == 'Clinic') & 
        (df['product_name'].isin([top_antecedent_clinic, top_consequent_clinic]))
    ]

    clinic_bundle_quantities = clinic_transactions_df.groupby(['transaction_id', 'product_name'])['quantity_sold'].sum().unstack()
    clinic_average_quantities = clinic_bundle_quantities.mean()

    return hospital_average_quantities, clinic_average_quantities
