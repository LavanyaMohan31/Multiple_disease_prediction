#streamlit run Emp_Att.py
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.figure_factory as ff
import joblib
from imblearn.pipeline import Pipeline as IMBPipeline
#-------------------------------------------------------------------------
# Load Data
sheet_id1="1GQ2wltXJ-veItC2BVoWJx7MmeS2mcljb62dVRrAMYZk"
sheet_name1="Sheet1"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id1}/gviz/tq?tqx=out:csv&sheet={sheet_name1}"
parkinson_df=pd.read_csv(url)
#------------------------------------------------------------------------
sheet_id2="1uiQfhlb0isYp2JefSVzIJOngN0viwsP0PzcMab-dUw8"
sheet_name2="Sheet1"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id2}/gviz/tq?tqx=out:csv&sheet={sheet_name2}"
liver_df=pd.read_csv(url)
#-----------------------------------------------------------------------
sheet_id3="1cew2eJA_0YRwYC5BOZdSl8fhTY0hSOVtD0xRfhevlgI"
sheet_name3="Sheet1"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id3}/gviz/tq?tqx=out:csv&sheet={sheet_name3}"
kidney_df=pd.read_csv(url)
#-------------------------------------------------------------------------
st.set_page_config(page_title="Multiple_Disease_Prediction",page_icon="👩🏻‍⚕️", layout="wide")

#sidebar
def show_tab(tab_name):
    st.session_state["active_tab"] = tab_name
if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = "Home"

# Custom CSS for equal-size buttons + text color
st.markdown("""
    <style>
    /* Center buttons inside sidebar */
    .stSidebar .element-container {
        display: flex;
        justify-content: center;
    }

    /* Style buttons */
    div.stButton > button {
        width: 170px;            /* Equal width */
        height: 50px;            /* Equal height */
        font-size: 24px;           /* Font size */
        color: white !important; /* Text color */
        background-color:#043369; /* Background color */
        border-radius: 8px;      /* Rounded corners */
        font-weight: bold;
        margin: 2px 0;          /* Space between buttons */
        
    }
    div.stButton > button:hover {
        background-color: #0779fa; /* Hover effect */
        color: #fff !important;
    }

    /* Add top margin to push buttons down */
    .stSidebar .stButton:first-child {
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)


# Sidebar buttons
with st.sidebar:
    st.button("🏨 Home", on_click=show_tab, args=("Home",))
    st.markdown(
    '<p style="color:#043369; font-size:22px; font-weight:bold;">🔍PREDICTION</p>',
    unsafe_allow_html=True
    )
    st.button("🧠 Parkinson ", on_click=show_tab, args=("Parkinson",))
    st.button("🩸 Liver ", on_click=show_tab, args=("Liver",))
    st.button("🫘 Kidney ", on_click=show_tab, args=("Kidney",))

# Lock sidebar width using CSS
st.markdown("""
    <style>
    /* Fix the sidebar width */
    section[data-testid="stSidebar"] {
        width: 200px !important;
        min-width: 200px !important;
        max-width: 200px !important;
        /*background-color: #e6f2ff;*/
        background: linear-gradient(#c0dbfa,#a7cefa,#6eabf0,#3f8ee8, #0779fa, #044187); /* Blue gradient */
        color: white; /* Text color */
    }
            
    /* Ensure all text inside sidebar inherits the color */
    section[data-testid="stSidebar"] * {
        color: white !important;
    }
    /* Fix the main content position accordingly */
    section.main {
        margin-left: 200px;
    }
  
    </style>
""", unsafe_allow_html=True)

#--------------------------------------------------------------------------------------------

# Sticky title with space for the sidebar

st.markdown("""
    <style>
    /* Fixed header below top navbar & beside sidebar */
    .fixed-title {
        position: fixed;
        top: 3.5rem;  /* Adjusted for Streamlit's top nav (approx 56px) */
        left: 12.5rem;  /* Sidebar width */
        width: calc(100% - 14rem);
        z-index: 9999;
        background-color: #044187;
        color: white;
        text-align: center;
        padding: 10px 0;
        font-size: 24px;
        font-weight: bold;
        border-bottom: 2px solid #043369;
    }

    /* Push content below fixed header */
    .spacer {
        height: 5px;
    }
    </style>

    <div class="fixed-title">
        👩🏻‍⚕️ MULTIPLE DISEASE PREDICTION
    </div>
    <div class="spacer"></div>
""", unsafe_allow_html=True)

#---------------------------------------------------------------------------------------------

#Styled Header
def styled_header(text, icon="📊"):
    st.markdown(f"""
        <style>
        .styled-h3 {{
            background-color: #9c025a !important;
            color: white !important;
            border: 2px solid #044187;
            padding: 4px 10px !important;
            border-radius: 5px;
            text-align: left;
            font-size: 18px !important;
            font-weight: 600 !important;
            margin-bottom: 10px !important;
            line-height: 1.2 !important;
            width: fit-content;
            margin-left: auto;
            margin-right: auto;
        }}
        </style>

        <h3 class="styled-h3">{icon} {text}</h3>
    """, unsafe_allow_html=True)
#-------------------------------------------------------------------------
def styled_title(text, color="#044187", size="24px", align="center"):
    st.markdown(
        f"""
        <h2 style='text-align: {align}; 
                   color: {color}; 
                   font-size: {size}; 
                   font-weight: bold;
                   text-shadow: 1px 1px 2px #fff;'>
            {text}
        </h2>
        """,
        unsafe_allow_html=True
    )

#---------------------------------------------------------------------------------------------
# Show selected tab content
if st.session_state["active_tab"] == "Home":
    st.markdown("""
        <h1 style='text-align: center; color: #044187; font-family: "Arial", sans-serif;'>
            Welcome
        </h1>
    """, unsafe_allow_html=True)

    # Centered, colored quote with padding
    st.markdown("""
        <p style='text-align: center; color: #ffffff; font-size: 20px; font-style: italic; 
                  margin-top: 5px; margin-bottom: 30px;'>
            "Take care of your body. It’s the only place you have to live, Health is real wealth"
        </p>
    """, unsafe_allow_html=True)
    st.image(
        "C:/DS_Programs/Project4_Disease_prediction/img.png",  # replace with your image path or URL
         width='stretch'
    )
    #--------------------------------------------------------------------------------------------------


elif st.session_state["active_tab"] == "Parkinson":
    
    
    model = joblib.load("C:/DS_Programs/Project4_Disease_prediction/parkinson_model.pkl")  

    st.markdown("""
        <h2 style='text-align: center; color: #044187; font-family: "Arial", sans-serif;'>
            PARKINSON PREDICTION
        </h2>
    """, unsafe_allow_html=True)
    styled_title("Enter Person Details")

    
    MDVP_Fo_Hz = st.number_input("MDVP:Fo(Hz)", min_value=0.000, max_value=300.000, value=120.000, format="%.3f")
    MDVP_Jitter_percent = st.number_input("MDVP:Jitter(%)", min_value=0.00000, max_value=1.00000, value=0.00500, format="%.5f")
    MDVP_Shimmer = st.number_input("MDVP:Shimmer", min_value=0.00000, max_value=1.00000, value=0.04000, format="%.5f")
    RPDE = st.number_input("RPDE", min_value=0.000000, max_value=1.000000, value=0.450000, format="%.6f")
    DFA = st.number_input("DFA", min_value=0.000000, max_value=1.000000, value=0.800000, format="%.6f")
    PPE = st.number_input("PPE", min_value=0.000000, max_value=1.000000, value=0.250000, format="%.6f")
    spread1 = st.number_input("spread1", min_value=-10.000000, max_value=10.000000, value=-4.500000, format="%.6f")
    spread2 = st.number_input("spread2", min_value=0.000000, max_value=10.000000, value=0.250000, format="%.6f")
    HNR = st.number_input("HNR", min_value=0.000, max_value=50.000, value=20.000, format="%.3f")


    input_data = pd.DataFrame([[
        MDVP_Fo_Hz, MDVP_Jitter_percent, MDVP_Shimmer, RPDE, DFA, PPE, spread1, spread2, HNR
    ]], columns=[
            'MDVP:Fo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Shimmer', 'RPDE', 'DFA', 'PPE', 'spread1', 'spread2', 'HNR'
    ])


   
    if st.button("🔍 Predict Parkinson’s Disease"):
        prediction= model.predict(input_data)[0]
        
        if prediction == 1:
            st.error(f"⚠️ The patient is likely to have Parkinson’s disease. ")
        else:
            st.success(f"✅ The patient is unlikely to have Parkinson’s disease.")
     #---------------------------------------------------------------------------------------------------

elif st.session_state["active_tab"] == "Liver":
    
    
    model = joblib.load("C:/DS_Programs/Project4_Disease_prediction/liver_model.pkl")  

    st.markdown("""
        <h2 style='text-align: center; color: #044187; font-family: "Arial", sans-serif;'>
            LIVER DISEASE PREDICTION
        </h2>
    """, unsafe_allow_html=True)
    styled_title("Enter Person Details")
    
    Age = st.number_input("🎂 Age", min_value=1, max_value=100, value=40, step=1)
    Total_Bilirubin = st.number_input("🧪 Total Bilirubin ", min_value=0.0, max_value=75.0, value=1.2, step=0.001)
    Direct_Bilirubin = st.number_input("🧫 Direct Bilirubin", min_value=0.0, max_value=20.0, value=0.3, step=0.001)
    Alkaline_Phosphotase = st.number_input("🧬 Alkaline Phosphotase", min_value=50, max_value=3000, value=200, step=1)
    Alamine_Aminotransferase = st.number_input("🧫 Alamine Aminotransferase", min_value=5, max_value=3000, value=30, step=1)
    Aspartate_Aminotransferase = st.number_input("🧫 Aspartate Aminotransferase", min_value=5, max_value=3000, value=35, step=1)
    Total_Protiens = st.number_input("💧 Total Proteins", min_value=2.0, max_value=10.0, value=6.8, step=0.001)
    Albumin = st.number_input("💧 Albumin", min_value=1.0, max_value=6.0, value=3.5, step=0.001)

    # --------------------------
    # Prepare input DataFrame
    # --------------------------
    input_data = pd.DataFrame([[
        Age, Total_Bilirubin, Direct_Bilirubin, Alkaline_Phosphotase,
        Alamine_Aminotransferase, Aspartate_Aminotransferase,
        Total_Protiens, Albumin
    ]], columns=[
        'Age', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase',
        'Alamine_Aminotransferase', 'Aspartate_Aminotransferase',
        'Total_Protiens', 'Albumin'
    ])

    if st.button("🔍 Predict Liver Disease"):
        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.error("⚠️ The patient is likely to have Liver Disease.")
        else:
            st.success("✅ The patient is unlikely to have Liver Disease.")


    #--------------------------------------------------------------------------------------------------
elif st.session_state["active_tab"] == "Kidney":
    
    
    model = joblib.load("C:/DS_Programs/Project4_Disease_prediction/KIDNEY_model.pkl")  

    st.markdown("""
        <h2 style='text-align: center; color: #044187; font-family: "Arial", sans-serif;'>
            KIDNEY DISEASE PREDICTION
        </h2>
    """, unsafe_allow_html=True)
    styled_title("Enter Person Details")  

   

    # Numeric inputs
    age = st.number_input("🎂 Age", min_value=1, max_value=120, value=45, step=1)
    bp = st.number_input("💓 Blood Pressure", min_value=50, max_value=200, value=80, step=1)
    sg = st.number_input("🧪 Specific Gravity", min_value=1.0, max_value=1.025, value=1.015, step=0.001)
    al = st.number_input("🧫 Albumin", min_value=0, max_value=5, value=1, step=1)
    su = st.number_input("🍬 Sugar", min_value=0, max_value=5, value=0, step=1)
    bgr = st.number_input("🩸 Blood Glucose Random", min_value=50, max_value=500, value=120, step=1)
    bu = st.number_input("💉 Blood Urea", min_value=1, max_value=300, value=40, step=1)
    sc = st.number_input("🧫 Serum Creatinine", min_value=0.1, max_value=15.0, value=1.2, step=0.01)
    hemo = st.number_input("🩸 Hemoglobin", min_value=3.0, max_value=20.0, value=13.0, step=0.1)
    pcv = st.number_input("💧 Packed Cell Volume", min_value=10, max_value=60, value=40, step=1)
    wc = st.number_input("🧬 White Blood Cell Count", min_value=2000, max_value=30000, value=8000, step=100)
    rc = st.number_input("🧬 Red Blood Cell Count", min_value=2.0, max_value=8.0, value=4.8, step=0.1)

    # Categorical inputs
    rbc = st.selectbox("🔴 Red Blood Cells", ["normal", "abnormal"])
    pc = st.selectbox("🧫 Pus Cell", ["normal", "abnormal"])
    pcc = st.selectbox("🧫 Pus Cell Clumps", ["present", "notpresent"])
    ba = st.selectbox("🦠 Bacteria", ["present", "notpresent"])
    htn = st.selectbox("💓 Hypertension", ["yes", "no"])
    dm = st.selectbox("🍬 Diabetes Mellitus", ["yes", "no"])
    appet = st.selectbox("🍽️ Appetite", ["good", "poor"])
    pe = st.selectbox("🦵 Pedal Edema", ["yes", "no"])
    ane = st.selectbox("💉 Anemia", ["yes", "no"])

    # Prepare input for model
    input_data = pd.DataFrame([[
        age, bp, sg, al, su, bgr, bu, sc, hemo, pcv, wc, rc,
        rbc, pc, pcc, ba, htn, dm, appet, pe, ane
    ]], columns=[
        'age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc',
        'hemo', 'pcv', 'wc', 'rc', 'rbc', 'pc', 'pcc', 'ba',
        'htn', 'dm', 'appet', 'pe', 'ane'
    ])

    # Prediction button
    if st.button("🔍 Predict Kidney Disease"):
        prediction = model.predict(input_data)[0]
        
        if prediction == 1:
            st.error("⚠️ The patient is likely to have **Kidney Disease.**")
           
        else:
            st.success("✅ The patient is **likely to be healthy.**")
           