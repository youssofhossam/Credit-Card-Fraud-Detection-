import streamlit as st
import joblib
import pandas as pd
import numpy as np

@st.cache_resource
def load_assets():
    data = joblib.load('final_fraud_model.pkl')
    return data['model'], data['scaler']

model, scaler = load_assets()
threshold = .5
test_samples = pd.read_csv("test_sample.csv")

st.title("🛡️ Credit Card Fraud Detection System")
st.write("This system uses a Random Forest Classifier with an optimized decision threshold to detect fraudulent transactions.")

st.sidebar.header("Simulation Settings")
input_option = st.sidebar.radio("Select Input Method:", ["Pick Random Sample", "Manual Input (Amount & Time only)"])

if input_option == "Pick Random Sample":
    st.subheader("Transaction Simulation")
    
    if st.button("🎲 Generate Random Transaction"):
        # Pick a random row from our saved test file
        random_row = test_samples.sample(1)
        
        # Save it to session state so it stays on screen
        st.session_state['current_row'] = random_row.drop(columns=['Class'])
        st.session_state['Class'] = random_row['Class'].values[0]

    # Display the transaction if one is selected
    if 'current_row' in st.session_state:
        row = st.session_state['current_row']
        
        # Show the data to the user
        st.write("### Transaction Details:")
        st.dataframe(row)
        
        # PREDICTION BUTTON
        if st.button("🔍 Analyze Transaction"):
            # 1. Scale the input
            # (Ensure columns match exactly what scaler expects)
            scaled_row = scaler.transform(row)
            
            # 2. Get Probability
            probability = model.predict_proba(scaled_row)[0][1]
            
            # 3. Apply Custom Threshold
            is_fraud = probability >= threshold
            
            # 4. Display Results
            st.divider()
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Fraud Probability", f"{probability:.2%}")
                
            with col2:
                if is_fraud:
                    st.error(f"🚨 FRAUD DETECTED")
                else:
                    st.success("✅ Transaction Legitimate")
            
            # Show "Why" (Simple Logic)
            st.write(f"**Decision Logic:** Probability ({probability:.2f}) > Threshold ({threshold})")
            
            # Reveal the Truth
            st.info(f"📝 **Actual Label:** {'Fraud' if st.session_state['Class'] == 1 else 'Legitimate'}")

elif input_option == "Manual Input (Amount & Time only)":
    st.subheader("Manual Transaction Simulation")
    st.info("ℹ️ The system picks a random background transaction (V1-V28), and you modify the Amount/Time.")

    # 1. Initialize Session State for the "Background Context"
    # This ensures the background doesn't change while you are typing.
    if 'manual_bg' not in st.session_state:
        st.session_state['manual_bg'] = test_samples.sample(1)

    # 2. Button to Shuffle Background
    if st.button("🎲 Pick New Random Background"):
        st.session_state['manual_bg'] = test_samples.sample(1)
        st.rerun() # Force reload to update input boxes

    # Get the current background row
    bg_row = st.session_state['manual_bg']
    
    # Show the user what the original class was (Hidden Hint)
    orig_label = "Fraud" if bg_row['Class'].values[0] == 1 else "Normal"
    st.caption(f"Current Background Context: **{orig_label} Transaction**")

    # 3. User Inputs (Defaults taken from the random row)
    # We use float() to convert numpy values to standard python floats
    default_time = float(bg_row['Time'].values[0])
    default_amount = float(bg_row['Amount'].values[0])
    
    col1, col2 = st.columns(2)
    with col1:
        new_time = st.number_input("Time (Seconds)", value=default_time)
    with col2:
        new_amount = st.number_input("Transaction Amount ($)", value=default_amount)

    # 4. Run Prediction
    if st.button("🔍 Check Modified Transaction"):
        # Create the Hybrid Row
        hybrid_row = bg_row.drop(columns=['Class']).copy()
        
        # Overwrite with user values
        hybrid_row['Time'] = new_time
        hybrid_row['Amount'] = new_amount
        
        # Predict
        scaled_row = scaler.transform(hybrid_row)
        probability = model.predict_proba(scaled_row)[0][1]
        is_fraud = probability >= threshold

        # Display Results
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Fraud Probability", f"{probability:.2%}")
        with c2:
            if is_fraud:
                st.error("🚨 FRAUD DETECTED")
            else:
                st.success("✅ Transaction Legitimate")
        
        st.write(f"**Analysis:** You applied a **${new_amount}** amount to a hidden **{orig_label}** pattern.")