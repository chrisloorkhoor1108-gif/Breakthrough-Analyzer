import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Breakthrough Curve Analyzer")

uploaded_file = st.file_uploader("Upload your breakthrough CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(df)

    x_col = st.selectbox("Select time column", df.columns)
    y_col = st.selectbox("Select concentration/signal column", df.columns)

    fig, ax = plt.subplots()
    ax.plot(df[x_col], df[y_col])
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title("Breakthrough Curve")

    st.pyplot(fig)
