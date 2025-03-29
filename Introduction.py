import warnings
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(layout='wide')


### Title Column: ###
_, title_col, _ = st.columns([1, 3, 1])

with title_col:
    st.title('Thyroid Cancer Diagnosis Prediction App:')


### Reading The Dataframe: ###
file_path = "thyroid_cancer_risk.csv"

@st.cache_data
def load_data():
    data = pd.read_csv(file_path)
    return data

df = load_data()


### Dataframe Info: ###
colDescriptionList = [
    "Age of the patient.", "Patient’s gender (Male/Female).", "Country of residence.", "Patient’s ethnic background.",
    "Whether the patient has a family history of thyroid cancer (Yes/No).", "History of radiation exposure (Yes/No).", "Presence of iodine deficiency (Yes/No).",
    "Whether the patient smokes (Yes/No).", "Whether the patient is obese (Yes/No).", "Whether the patient has diabetes (Yes/No).",
    "Thyroid-Stimulating Hormone level (µIU/mL).", "Triiodothyronine level (ng/dL).", "Thyroxine level (µg/dL).", "Size of thyroid nodules (cm).",
    "Estimated risk of thyroid cancer (Low/Medium/High).", "Final diagnosis (Benign/Malignant)."
]

dfInfo = pd.DataFrame(
    {
        'Column Name' : [col for col in df.columns],
        'Non-null Count' : [(len(df) - df[col].isnull().sum()) for col in df.columns],
        'Data Type' : [str(df[col].dtype) for col in df.columns],
        'Column Description' : colDescriptionList
    }
)

_, dfInfo_col, _ = st.columns([1, 3, 1])

with dfInfo_col:
    st.write(
        "Hello and welcome to the **Thyroid Cancer Diagnosis Prediction App**. The dataset incorporates **212,691** statistics related to thyroid cancer risk factors.\
        It includes demographic facts, clinical history, lifestyle, factors, and key thyroid hormone degrees to assess the probability of thyroid most cancers.\
        \n\nSome basic info about the dataset is given in the table below:"
    )
    st.dataframe(dfInfo, height=595)

st.markdown("[Link to Dataset](https://www.kaggle.com/datasets/mzohaibzeeshan/thyroid-cancer-risk-dataset)")

if st.toggle(label="**Show Raw Data:**"):
    st.dataframe(df, height=600)