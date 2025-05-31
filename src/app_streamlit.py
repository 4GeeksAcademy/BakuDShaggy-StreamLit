import streamlit as st
import joblib
import pandas as pd
import os
import re

st.set_page_config(page_title="Titanic Survival Predictor", layout="wide")
st.title("🚢 Titanic Survival Predictor")
st.markdown("""Predict whether a passenger would have survived the Titanic disaster based on their characteristics. 
*Model accuracy: 78.21%*
""")
@st.cache_resource
def load_model():
    # Get current directory of this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Go up one level to project root
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    
    # Construct model path
    model_path = os.path.join(project_root, 'models', 'titanic_rf_model.joblib')
    
    # Debug info (shows in Render logs)
    print(f"Current directory: {current_dir}")
    print(f"Project root: {project_root}")
    print(f"Model path: {model_path}")
    
    return joblib.load(model_path)

model = load_model()
with st.form("passenger_form"):
    st.header("Passenger Details")
    
    # Two columns layout
    col1, col2 = st.columns(2)
    
    with col1:
        sex = st.radio("Gender", ["Male", "Female"], index=0)
        age = st.slider("Age", 0, 100, 28, help="Age of passenger in years")
        pclass = st.selectbox("Passenger Class", 
                             ["First (1)", "Second (2)", "Third (3)"], 
                             index=2)
    
    with col2:
        sibsp = st.number_input("Siblings/Spouses Aboard", 
                               min_value=0, max_value=10, value=0)
        parch = st.number_input("Parents/Children Aboard", 
                              min_value=0, max_value=10, value=0)
        fare = st.number_input("Fare (£)", 
                             min_value=0.0, max_value=1000.0, 
                             value=14.45, step=1.0,
                             help="Ticket price in British pounds")
    
    
    submit_button = st.form_submit_button("Predict Survival", type="primary")

if submit_button:
    sex_num = 0 if sex == "Male" else 1
    pclass_num = int(re.search(r'\((\d+)\)', pclass).group(1))
    features = pd.DataFrame([[pclass_num, sex_num, age, sibsp, parch, fare]],
                           columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare'])
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][prediction]

    if prediction == 1:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/RMS_Titanic_3.jpg/800px-RMS_Titanic_3.jpg", 
             width=400)
    else:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/48/Titanic_plans_1911.jpg/800px-Titanic_plans_1911.jpg",
             width=400)


st.info("""
        **Key factors:**
        - Gender was the strongest predictor (Ladies First)
        - Higher Classes but I am not sure why would it be, I only remember it was not the usual reasoning)
        - Children under 10 had more survival rate
        - It was harder for Larger families to evacuate is my only reasoning
        """)



st.markdown("---")
st.caption("Built with Streamlit | MOdel: Random Forest | Data: List of passenger of the Titanic ")
        