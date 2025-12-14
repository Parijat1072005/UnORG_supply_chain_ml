import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
import os

app = Flask(__name__)

# --- Configuration ---
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

def load_data():
    """Helper to load datasets safely"""
    try:
        # Load Probabilities
        prob_df = pd.read_csv(os.path.join(DATA_DIR, 'order_probability_next_14_days.csv'))
        
        # Load Recommendations
        rec_df = pd.read_csv(os.path.join(DATA_DIR, 'customer_recommendations.csv'))
        
        # Load Inventory
        inv_df = pd.read_csv(os.path.join(DATA_DIR, 'inventory_plan.csv'))
        
        return prob_df, rec_df, inv_df
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure CSV files are in the 'data' folder.")
        return None, None, None

@app.route('/')
def dashboard():
    prob_df, rec_df, inv_df = load_data()
    if prob_df is None:
        return "Data files not found. Please run your ML Notebook first."

    # --- KPI Calculations ---
    total_customers = prob_df['customer_id'].nunique()
    
    # Customers with > 50% probability to order
    likely_buyers = prob_df[prob_df['order_probability'] > 0.5].shape[0]
    
    # Products needing restock (Recommended Order > 0)
    critical_stock_items = inv_df[inv_df['recommended_order'] > 0].shape[0]
    
    # Top 5 items by demand
    top_items = inv_df.sort_values(by='14_day_demand', ascending=False).head(5)[['item_name', '14_day_demand']].to_dict(orient='records')

    return render_template('dashboard.html', 
                           total_customers=total_customers,
                           likely_buyers=likely_buyers,
                           critical_stock_items=critical_stock_items,
                           top_items=top_items)

@app.route('/customer_insight', methods=['GET', 'POST'])
def customer_insight():
    prob_df, rec_df, _ = load_data()
    
    search_result = None
    customer_id = None
    error_msg = None

    if request.method == 'POST':
        customer_id = request.form.get('customer_id')
        
        # Try to convert to int if your IDs are integers
        try:
            customer_id = int(customer_id)
        except ValueError:
            pass # Keep as string if conversion fails

        # Check if customer exists in probability data
        customer_prob = prob_df[prob_df['customer_id'] == customer_id]
        
        if not customer_prob.empty:
            # Get Probability
            probability = round(customer_prob.iloc[0]['order_probability'] * 100, 2)
            
            # Get Recommendations
            recs = rec_df[rec_df['customer_id'] == customer_id]
            rec_list = recs[['item_name', 'predicted_quantity', 'selection_probability']].to_dict(orient='records')
            
            search_result = {
                'id': customer_id,
                'probability': probability,
                'recommendations': rec_list
            }
        else:
            error_msg = "Customer ID not found in predictions."

    return render_template('customer_search.html', result=search_result, error=error_msg)

@app.route('/inventory')
def inventory():
    _, _, inv_df = load_data()
    
    # Convert entire dataframe to list of dicts for the HTML table
    inventory_data = inv_df.to_dict(orient='records')
    
    return render_template('inventory.html', inventory=inventory_data)

if __name__ == '__main__':
    app.run(debug=True, port=5000)