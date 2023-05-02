import streamlit as st
import time
import numpy as np
import pandas as pd
import plotly.express as px
from progress_bar import InitBar
from tqdm import tqdm
from time import sleep

st.set_page_config(page_title="Historical", page_icon="📈")

st.markdown("# Historical Plotting")
st.image('picture/893216.png', width=45)

st.write("""This page illustrates a historical dataset from sensor.""")

#read csv file
input = pd.read_csv('./01.Raw Data/Set1 CUPS Result (For analysis).csv')
input

#CUPS Training plot
st.subheader('Bar chart of Support Type')
Support_Type = px.line(input,x='Finding No.', y=['Support Type'],title='Finding No. vs. Support Type')
st.line_chart(Support_Type)

#----------------------------------------------------------------
st.sidebar.header("Loading...")
progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()
for i in range(1, 101):
        #status_text.text("%i%% Complete" % i)
        progress_bar.progress(i)
        time.sleep(0.03)



# rerun.
st.button("Re-run")

for i in range(1, 101):
        status_text.text("%i%% Complete" % i)
        time.sleep(0.02)
