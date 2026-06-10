import os
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from PIL import Image

base_dir = os.path.dirname(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
model_path = os.path.join(base_dir, "Model", "catboost_pipeline.pkl")
image_path = os.path.join(base_dir, "Image")
data_path = os.path.join(base_dir, "Data")

# Load model
model = joblib.load(model_path)

# Feature Engineering
def create_features(df):
    df = df.copy()
    df['contacted_before'] = np.where(df['previous'] > 0,1,0)
    df['total_loans'] = (
                            (df['housing'] == 'yes').astype(int)+
                            (df['loan'] == 'yes').astype(int)
                        )
    df['age_balance'] = (df['age']*df['balance'])
    return df

st.set_page_config(page_title="Bank Term Deposit Prediction", layout="wide")
st.markdown("""
    <style>
           .block-container {
                padding-top: 1rem;
                padding-bottom: 0rem;
                padding-left: 3rem;
                padding-right: 3rem;
            }
    </style>
    """, unsafe_allow_html=True)

# change in form height & width
st.html(
    """
    <style>
    div[data-testid="stForm"] {
        height: 400px;
        width: 800px;
        overflow-y: auto;
    }
    </style>
    """
)


st.title("Bank Direct Marketing", text_alignment="center")

st.markdown("The Bank's marketing efforts relied on direct phone-based communication. " \
                "To effectively determine a customer's interest in subscribing to a term deposit, " \
                "repeated contacts with the same client were often required.")
tab1, tab2, tab3 = st.tabs(
    [
        "**Bank Term Deposit Subscription Prediction**",
        "**Model Performance Leaderboard**",
        "**Visualization**"
    ]
)
# ------------------------
# User Inputs
# ------------------------
with tab1:
    left_column, middle_column, right_column = st.columns([1, 2, 1])
       
    with middle_column:
        st.markdown("Complete the form & click **Predict** to check whether the client is likely to subscribe to a Term Deposit.")
     
        with st.form(key="depositprediction_form"):
            col1, col2 = st.columns(2)

            with col1:
                age = st.number_input("**Age**", 18, 100, 35)                    
                education = st.selectbox("**Education**",['primary', 'secondary', 'tertiary', 'unknown'])
                housing = st.selectbox("**Housing Loan**",['yes', 'no'])
                balance = st.number_input("**Account Balance**",value=1000)
                day = st.slider("**Last Contact Day**", 1, 31, 15)
                campaign = st.number_input("**Number of Contacts During Campaign**",value=1)
                previous = st.number_input("**Previous Contacts**",value=0)
                poutcome = st.selectbox("**Previous Campaign Outcome**",['success','failure','other','unknown'])
                
            with col2:
                marital = st.selectbox("**Marital Status**",['married', 'single', 'divorced'])
                job = st.selectbox(
                                    "**Job**",['admin.', 'blue-collar', 'entrepreneur','housemaid', 'management', 'retired',
                                            'self-employed', 'services','student', 'technician', 'unemployed','unknown']
                                )            
                loan = st.selectbox("**Personal Loan**",['yes', 'no'])
                default = st.selectbox("**Credit Default**",['yes', 'no'])
                month = st.selectbox("**Last Contact Month**",['jan','feb','mar','apr','may','jun',
                                                            'jul','aug','sep','oct','nov','dec']
                                    )
                contact = st.selectbox("**Contact Type**",['cellular', 'telephone'])
                pdays = st.number_input("**Days Since Last Contact**",value=999)      

            submit_button = st.form_submit_button(label='Predict', use_container_width=True, type="primary")
        # ------------------------
        # Prediction
        # ------------------------

        if submit_button:
            customer = pd.DataFrame({
                                        'age':[age],
                                        'job':[job],
                                        'marital':[marital],
                                        'education':[education],
                                        'default':[default],
                                        'balance':[balance],
                                        'housing':[housing],
                                        'loan':[loan],
                                        'contact':[contact],
                                        'day':[day],
                                        'month':[month],
                                        'campaign':[campaign],
                                        'pdays':[pdays],
                                        'previous':[previous],
                                        'poutcome':[poutcome]
                                    })
            customer_df = create_features(customer)

            # Predict
            prediction = model.predict(customer_df)[0]
            probability = model.predict_proba(customer_df)[0]

            if prediction == 1:
                st.success("✅ Customer is likely to subscribe.")
            else:
                st.error("❌ Customer is unlikely to subscribe.") 

with tab2:
    st.markdown("The following table compares the performance of all machine learning "
                  "and deep learning models evaluated for predicting bank term deposit subscriptions.")  

    model_data = {
                    "**Rank**" : ["🥇 1","🥈 2","🥉 3","4","5","6","7","8","9"],
                    "**Model**" : ["CatBoost (Threshold Optimized)","LightGBM (Tuned)",
                               "XGBoost (Tuned)","Gradient Boosting","Random Forest","Logistic Regression",
                               "Gaussian Naive Bayes","Deep Neural Network (DNN)","Artificial Neural Network (ANN)"
                                ],
                    "**Accuracy**" : ["89.09%","89.22%","89.36%","88.36%","88.82%","75.30%","76.24%","-","-"],
                    "**Precision**": ["54.40%","57.14%","59.52%","50.31%","53.88%","26.35%","26.82%","19.69%","19.45%"],
                    "**Recall**": ["41.49%","31.38%","28.36%","38.47%","30.81%","61.91%","59.64%","81.10%","82.42%"],
                    "**F1 Score**" : ["47.08%","40.51%","38.41%","43.60%","39.21%","36.96%","37.00%","31.68%","31.47%"],
                    "**ROC-AUC**" : ["0.795","0.795","0.79","0.78","0.78","0.766","0.744","0.783","0.788"]
                }
    st.table(model_data, border="horizontal")

with tab3:
    st.subheader("Important Features", text_alignment="left")
    image = Image.open(fr"{image_path}\ImportantFeature.png")
    st.write(' ')
    st.write(' ')
    st.image(image)

    st.subheader("SHAP Plot", text_alignment="left")
    image = Image.open(fr"{image_path}\SHAP.png")
    st.write(' ')
    st.write(' ')
    st.image(image)

    st.subheader("ROC Curve", text_alignment="left")
    image = Image.open(fr"{image_path}\ROC_Curve.png")
    st.write(' ')
    st.write(' ')
    st.image(image)

    st.subheader("Confusion Matrix", text_alignment="left")
    image = Image.open(fr"{image_path}\Confusion_Matrix.png")
    st.write(' ')
    st.write(' ')
    st.image(image)