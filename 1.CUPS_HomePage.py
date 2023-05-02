import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

#import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Homepage",page_icon="")

st.sidebar.success("Select a page above.")

st.write('# CUPS Prediction')  #st.title('')
st.image('picture/maintenance.png', width=70)
st.markdown(
    """
    **This is a dashboard showing the CUPS Result**
""") 
st.markdown(
    """
    **👈 Select a demo dataset from the sidebar**
"""
)

st.image('picture/GCME_Logo.png')


st.sidebar.header("Input features for simulation")


# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

#select = st.sidebar.radio("Select Model",('XGBoost Regression','Voting Regressor'))
select = st.sidebar.radio("Select Model",('Random Forest','Logistic Regression'))

if uploaded_file is not None:
    input_df2 = pd.read_csv(uploaded_file)
    input_df = input_df2.drop(columns=['Finding No.'])
else:
    def user_input_features():
        Support_Type = st.sidebar.selectbox('Support Type', options=['Bolted skid','Guide,Pipework resting on beam',
                                                                 'Saddle support','Welded skid,Saddle support+guide',
                                                                 'Saddle support+shoe','Clamp+Guide','Clamp','Trunnion'])
        Insulation = st.sidebar.selectbox('Insulation', options=['Yes','No'])
        Fluid = st.sidebar.selectbox('Fluid', options=['Methyl Methacrylate','Sulfuric acid','Ammonia','Propylene','Hexane',
                                                   'Acrylonitrile','Liquefied Petroleum Gas'])
        Operating_Temp_min = st.sidebar.slider('Operating Temperature (minimum)',-30, 50,25)
        Operating_Temp_max = st.sidebar.slider('Operating Temperature (maximum)',0, 50,25)
        Material = st.sidebar.selectbox('Material', options=['Carbon Steel','Stainless Steel'])
        Coating = st.sidebar.selectbox('Coating', options=['Yes','No'])
        Primer = st.sidebar.selectbox('Primer', options=['Inorganic zinc Self-cure Solvent base'])
        Mist = st.sidebar.selectbox('Mist', options=['High-built Polyamide Epoxy'])
        Intermediate = st.sidebar.selectbox('Intermediate', options=['High-build Polyamide Epoxy'])
        Finish = st.sidebar.selectbox('Finish', options=['Aliphatic Polyurethane Final coat'])
        Anti_Corrosion = st.sidebar.selectbox('Anti-Corrosion', options=['Yes','No'])
        Environment = st.sidebar.selectbox('Environment', options=['Marine'])
        
        data = {
            'Support Type':Support_Type,
            'Insulation':Insulation,
            'Fluid':Fluid,
            'Operating Temperature (minimum)':Operating_Temp_min,
            'Operating Temperature (maximum)':Operating_Temp_max,
            'Material':Material,
            'Coating':Coating,
            'Primer':Primer,
            'Mist':Mist,
            'Intermediate':Intermediate,
            'Finish':Finish,
            'Anti-Corrosion':Anti_Corrosion,
            'Environment':Environment,
                }
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()


#-----------------------------------------------------------------
lastrow = len(input_df.index)

data_train = pd.read_csv('./01.Raw Data/Set1 CUPS Result (For analysis).csv')
raw_data = data_train.drop(columns=['Finding No.','Severity','Susceptibility'])


#st.table(input_df)
df = pd.concat([input_df,raw_data],axis=0,ignore_index=True)


# Selects only the first row (the user input data)
df = df[:] 
#st.table(df)


# Displays the user input features
st.subheader('1. Features for Simulation')
st.image('picture/simulation.png', width=45)

if uploaded_file is not None:
   st.write(input_df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using input parameters (shown below).')
    st.write(input_df)

#Create a function for LabelEncoder
def Encoder(df):
          columnsToEncode = list(df.select_dtypes(include=['category','object']))
          le = LabelEncoder()
          for feature in columnsToEncode:
              try:
                  df[feature] = le.fit_transform(df[feature])
              except:
                  print('Error encoding '+feature)
          return df
columnsToEncode = list(df.select_dtypes(include=['category','object']))
le = LabelEncoder()
for feature in columnsToEncode:
              try:
                  df[feature] = le.fit_transform(df[feature])
              except:
                  print('Error encoding '+feature)
df = Encoder(df)

# Scale data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df = scaler.fit_transform(df)
#st.write(df)

# Reads in saved regression model
load_clf1 = pickle.load(open('20220427_RFModel.pkl', 'rb'))
load_clf2 = pickle.load(open('20220427_LRModel.pkl', 'rb'))


# Apply model to make predictions
predict = pd.DataFrame(df).iloc[:lastrow]
#prediction = load_clf2.predict(predict)
if select == 'Random Forest':
    prediction = load_clf1.predict(predict)
elif select =='Logistic Regression':
    prediction = load_clf2.predict(predict)

#----------------------------------------------------------

st.subheader('2. CUPS Prediction')
st.image('picture/predictive-chart.png', width=45)
#st.write([prediction])
st.write('Severity')
st.write(prediction)
#-----------------------------------------------------------



st.subheader('3. Severity chart')
st.write("3.1 Bar chart of severity")
#line_fig = px.line(uploaded_file,x='Start_time', y='prediction', title='Line chart of eeverity')
#st.plotly_chart(line_fig)
st.bar_chart(prediction)

#st.write("3.2 Bar chart of severity")
#st.bar_chart(prediction)


#----------------------------------------------------------------
#Save prediction file to csv

predic = pd.Series(prediction, name='Severity')

df_final = pd.concat([input_df.iloc[:,:14], pd.Series(predic)], axis=1)

@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

csv = convert_df(df_final)


st.download_button(
    label="Download prediction as CSV",
    data=csv,
    file_name='prediction_file.csv',
    mime='text/csv',
)

#----------------------------------------------------------
