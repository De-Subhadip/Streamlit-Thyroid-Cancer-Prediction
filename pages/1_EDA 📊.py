import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

warnings.filterwarnings("ignore")

st.set_page_config(layout='wide')

_, title_col, _ = st.columns([1.5, 3, 1.5])

with title_col:
    st.title('Exploratory Data Analysis :bar_chart::')

with st.container(height=5, border=False):
    st.empty()

filePath = 'thyroid_cancer_risk.csv'

@st.cache_data
def load_data():
    data = pd.read_csv(filePath)
    return data

df = load_data()

catCols = [col for col in df.columns if df[col].dtype == object]
numCols = [col for col in df.columns if col not in catCols]

_, summary_col, _ = st.columns([1, 5, 1])

with summary_col:
    dfSummary = df[numCols].describe()
    st.subheader('Summary Statistics of the Dataframe:')
    st.dataframe(dfSummary)

with st.container(height=10, border=False):
    st.empty()


tab1, tab2, tab3, tab4 = st.tabs(["Histogram", "Box Plot", "Count Plot", "Pie Chart"])

with tab1:
    _, col_plot, _ = st.columns([1.5, 3, 1.5])
    with col_plot:
        _, histHeader, _ = st.columns(3)

        with histHeader:
            st.header(':red[Histogram:]')

        a, b, _ = st.columns(3)

        with a:
            histCol = st.selectbox(label = ':red-background[**Choose a column for Histogram:**]', options = numCols)

        with b:
            histColor = st.selectbox(label = ":red-background[**Column for coloring the histogram:**]", options = [None] + catCols)
        
        bins = st.slider(label=':red[Select the number of bins:]', min_value=10, max_value=1000, value=100, step=1)

        fig1 = px.histogram(df, x=histCol, nbins=bins, color=histColor, color_discrete_sequence=['indianred'] if histColor is None else None)
        fig1.update_layout(legend=dict(x=0.5, y=1.2, xanchor='center', yanchor='top', orientation="h"), width=800, height=600)
        histogram = st.plotly_chart(fig1)


with tab2:
    _, col_plot, _ = st.columns([1.5, 3, 1.5])
    with col_plot:
        _, boxHeader, _ = st.columns(3)

        with boxHeader:
            st.header(":blue[Box Plot:]")

        c, d, e = st.columns(3)

        with c:
            xAxisBox = st.selectbox(label = ":blue-background[**Column for Y-axis:**]", options=[None] + catCols)
        
        with d:
            yAxisBox = st.selectbox(label = ":blue-background[**Column for X-axis:**]", options=numCols)
        
        with e:
            boxColor = st.selectbox(label = ":blue-background[**Column for coloring the Box Plot:**]", options=[None] + catCols)
        
        fig2 = px.box(df, y=xAxisBox, x=yAxisBox, color=boxColor)
        fig2.update_layout(legend=dict(x=0.5, y=1.2, xanchor='center', yanchor='top', orientation="h"), height=625)
        boxPlot = st.plotly_chart(fig2)


with tab3:
    _, col_plot, _ = st.columns([1.5, 3, 1.5])
    with col_plot:
        _, countHeader, _ = st.columns(3)

        with countHeader:
            st.header(':red[Count Plot:]')

        f, _, _ = st.columns(3)

        with f:
            colCountPlot = st.selectbox(label = ":red-background[**Column for Count Plot:**]", options = catCols)

        dictCount1 = dict(df[colCountPlot].value_counts())
        dfCount1 = pd.DataFrame(
            {
                f"{colCountPlot}" : dictCount1.keys(),
                "Count" : dictCount1.values()
            }
        )

        fig3 = px.bar(dfCount1, x=f"{colCountPlot}", y='Count', color=f"{colCountPlot}", text_auto=True, color_discrete_sequence=px.colors.sequential.RdBu)
        fig3.update_layout(legend=dict(x=0.5, y=1.2, xanchor='center', yanchor='top', orientation="h"), height=625)
        countPlot = st.plotly_chart(fig3)


with tab4:
    _, col_plot, _ = st.columns([1.5, 3, 1.5])
    with col_plot:
        _, pieHeader, _ = st.columns(3)

        with pieHeader:
            st.header(":blue[Pie Chart:]")

        g, h, _ = st.columns(3)

        with g:
            colPie = st.selectbox(label = ":blue-background[**Column for Pie Chart:**]", options = catCols)
        
        with h:
            valPie = st.selectbox(label = f":blue-background[**Choose a value for {colPie}:**]", options = df[colPie].unique())

        dfPie = df[(df[colPie] == valPie)].drop([col for col in df.columns if col != 'Diagnosis'], axis=1)
        countDict = dict(dfPie['Diagnosis'].value_counts())
        countDF = pd.DataFrame(
            {
                "Diagnosis" : countDict.keys(),
                "Count" : countDict.values()
            }
        )

        fig4 = px.pie(countDF, names='Diagnosis', values='Count')
        fig4.update_layout(legend=dict(x=0.5, y=1.2, xanchor='center', yanchor='top', orientation="h"), height=625)
        st.plotly_chart(fig4)