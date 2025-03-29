import joblib
import warnings
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(layout='wide')

if 'disabled' not in st.session_state:
    st.session_state.disabled = True
    st.session_state.toggle_state = False
    st.session_state.predText = " "
    st.session_state.predProbText = " "



### Introduction: ###
_, title_col, _ = st.columns([1.5, 3, 1.5])

with title_col:
    st.title('Predicting Thyroid Cancer Type:')

with st.container(height=10, border=False):
    st.empty()

st.write(
    "The model used for predicting the diagnosis is a **Logistic Regression** model with all the default parameter values. Grid Search was performed\
    on the dataset with 20 cross-validation sets to find the parameters that yielded the highest accuracy. The numeric columns were scaled using a\
    **MinMaxScaler**, and the categorical columns were **one-hot encoded**. **Ordinal encoding** was used on the ***Thyroid_Cancer_Risk*** columns as it contained\
    the following values: ***Low***, ***Medium***, ***High***."
)



### Loading Files: ###
filePath = 'thyroid_cancer_risk.csv'

@st.cache_data
def load_data():
    data = pd.read_csv(filePath)
    return data
df = load_data()

@st.cache_data
def load_scaler():
    scaler = joblib.load('MinMaxScaler.pkl')
    return scaler
MinMaxScaler = load_scaler()

@st.cache_data
def load_LRmodel():
    model = joblib.load('Logistic Regression.pkl')
    return model
LR = load_LRmodel()



### Cleaning, Scaling, Encoding:###
df = df.drop(['Gender', 'Country', 'Ethnicity'], axis=1)
df['Age'] = df['Age'].astype(np.int8)
df[['TSH_Level', 'T3_Level', 'T4_Level', 'Nodule_Size']] = df[['TSH_Level', 'T3_Level', 'T4_Level', 'Nodule_Size']].astype(np.float16)

catCols2 = [col for col in df.columns if df[col].dtype == object and col != 'Thyroid_Cancer_Risk']
numCols = [col for col in df.columns if df[col].dtype != object]

mapDict = dict()
thyroidMapping = {
    "Low":0,
    "Medium":1,
    "High":2
}

def encode_scale(dataframe):
    for col in dataframe.columns:
        if col in catCols2:

            if col not in mapDict.keys():
                vals = dataframe[col].value_counts().index
                mapDict[col] = {vals[0]: 0, vals[1]: 1}
                dataframe[col] = dataframe[col].map(mapDict[col])
            else:
                dataframe[col] = dataframe[col].map(mapDict[col])
    
    dataframe['Thyroid_Cancer_Risk'] = dataframe['Thyroid_Cancer_Risk'].map(thyroidMapping)
    
    dataframe[numCols] = MinMaxScaler.transform(dataframe[numCols])
        
    return dataframe

df = encode_scale(df)


try:
    ### Input Widgets: ###
    with st.container(height=10, border=False):
        st.empty()

    predDict = {}

    def input_change():
        st.session_state.predText = " "
        st.session_state.toggle_state = False
        st.session_state.disabled = True

    _, input_col, _ = st.columns([1.5, 3, 1.5])

    with input_col:
        st.subheader("Select Values for Prediction:")

        col1, col2, col3 = st.columns(3)

        with col1:
            familyHistory = st.pills(label=':red[**Family History of Thyroid Cancer?**]', options=['Yes', 'No'], key='Family_History', default="No", on_change=input_change)
            predDict['Family_History'] = st.session_state.Family_History
        
        with col2:
            radiationExposure = st.pills(label=':red[**Previous Radiation Exposure?**]', options=['Yes', 'No'], key='Radiation_Exposure', default="No", on_change=input_change)
            predDict['Radiation_Exposure'] = st.session_state.Radiation_Exposure
        
        with col3:
            iodineDeficiency = st.pills(label=':red[**Suffering from Iodine Deficiency?**]', options=['Yes', 'No'], key='Iodine_Deficiency', default="No", on_change=input_change)
            predDict['Iodine_Deficiency'] = st.session_state.Iodine_Deficiency
        
        col4, col5, col6 = st.columns(3)

        with col4:
            smoking = st.pills(label=':red[**Do you Smoke?**]', options=["Yes", "No"], key="Smoking", default="No", on_change=input_change)
            predDict['Smoking'] = st.session_state.Smoking
        
        with col5:
            obesity = st.pills(label=":red[**Suffering from Obesity?**]", options=["Yes", "No"], key="Obesity", default="No", on_change=input_change)
            predDict['Obesity'] = st.session_state.Obesity
        
        with col6:
            diabetes = st.pills(label=":red[**Suffering from Diabetes?**]", options=["Yes", "No"], key='Diabetes', default="No", on_change=input_change)
            predDict['Diabetes'] = st.session_state.Diabetes

        col7, col8, col9 = st.columns(3)

        with col7:
            TSHLevel = st.number_input(label=":red[**Enter TSH Level:**]", min_value=0.0, max_value=10.0, step=0.01, key='TSH_Level', on_change=input_change)
            predDict['TSH_Level'] = st.session_state.TSH_Level

        with col8:
            T3Level = st.number_input(label=":red[**Enter T3 Level:**]", min_value=0.0, max_value=5.0, step=0.01, key='T3_Level', on_change=input_change)
            predDict['T3_Level'] = st.session_state.T3_Level

        with col9:
            T4Level = st.number_input(label=":red[**Enter T4 Level:**]", min_value=4.0, max_value=12.0, step=0.01, key='T4_Level', on_change=input_change)
            predDict['T4_Level'] = st.session_state.T4_Level
        
        thyroidCancerRisk = st.select_slider(label=":red[**Select the Risk of Thyroid Cancer:**]", options=['Low', 'Medium', 'High'], key='Thyroid_Cancer_Risk', on_change=input_change)
        predDict['Thyroid_Cancer_Risk'] = st.session_state.Thyroid_Cancer_Risk

        age = st.slider(label=":red[**Select your Age:**]", min_value=15, max_value=90, step=1, key="Age", on_change=input_change)
        predDict['Age'] = st.session_state.Age

        nodule = st.slider(label=":red[**Select Nodule Size:**]", min_value=0.0, max_value=5.0, step=0.01, key="Nodule_Size", on_change=input_change)
        predDict['Nodule_Size'] = st.session_state.Nodule_Size



    ### Prediction: ###
    dfPred = pd.DataFrame([predDict.values()], columns=list(predDict.keys()))
    dfPred = encode_scale(dfPred)
    dfPred = dfPred[[col for col in df.columns if col!= 'Diagnosis']]

    pred = LR.predict(dfPred)
    predProb = LR.predict_proba(dfPred)
    n = np.argmax(predProb, axis=1)
    chance = predProb[0][n]
    chance = np.round(chance, 3)



    ### Prediction Button: ###
    with st.container(height=10, border=False):
        st.empty()

    def button_click():
        st.session_state.disabled = False
        st.session_state.toggle_state = False

        if pred == 1:
            st.session_state.predText = ":red[Thyroid is Malignant]:warning:"
            st.session_state.predProbText = f":red[**Probability of being malignant:**] **:red{chance}**"
        else:
            st.session_state.predText = ":green[Thyroid is Benign]:large_green_circle:"
            st.session_state.predProbText = f":green[**Probability of being Benign:**] **:green{chance}**"


    _, _, _, button_col, _, _, _ = st.columns(7)

    with button_col:
        button = st.button(label="**Predict Diagnosis:**", on_click=button_click)



    ### Showing Prediction: ###
    with st.container(height=10, border=False):
        st.empty()

    _, pred_col, _ = st.columns(3)

    with pred_col:
        with st.container(height=250, border=False):
            with st.container(height=75, border=True):
                st.subheader(st.session_state.predText)

            toggle = st.toggle(label="**Show Prediction Probability:**", disabled=st.session_state.disabled, key='toggle_state')

            if toggle:
                st.write(st.session_state.predProbText)

except Exception:
    st.error("ERROR!\n\nPlease select all the inputs.")