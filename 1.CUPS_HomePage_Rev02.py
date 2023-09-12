import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
#import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

#from matplotlib import pyplot as plt


st.set_page_config(page_title="Homepage",page_icon="")

st.sidebar.success("Select a page above.")

st.write('# CUPS Prediction')  #st.title('')
st.image('picture/check.png', width=50)
st.markdown(
    """
    **This is a dashboard showing the CUPS Result**
""") 
st.markdown(
    """
    **👈 Select a demo dataset from the sidebar**
"""
)

st.image('picture/GCME_Logo.png', width=250)


st.sidebar.header("Input features for simulation")


# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])


select = st.sidebar.radio("Select Model",('Random Forest','Decision Tree'))

if uploaded_file is not None:
    input_df2 = pd.read_csv(uploaded_file)
    input_df = input_df2.drop(columns=['Finding No.'])
else:
    def user_input_features():
        Support_Type = st.sidebar.selectbox('Support Type', options=['U-bolt','Pipework resting on beam','Bolted skid','Guide',
                                                                     'Pipework resting on beam/partially welded saddle','Trunnion',
                                                                     'Saddle support','Welded skid','Pipework resting on beam',
                                                                     'Trunnion','Saddle support','Welded skid','pipework resting on beam',
                                                                     'Saddle support+guide','Pipework resting on beam/Guide/partially welded saddle',
                                                                     'Saddle support+shoe','Pipework resting on beam/Guide',
                                                                     'Pipework resting on beam/fully welded saddle ','Welded skid/anchor',
                                                                     'Riser Clamp','pipework resting on beam/guide',
                                                                     'Clamp+Guide','Clamp','Saddle support – loose saddle','Hanger',
                                                                     'Anchor/pipework resting on beam','Pipework resting on beam', 
                                                                     'Pipework resting on beam/Guide/ partially welded saddle','Anchor'])
        Insulation = st.sidebar.selectbox('Insulation', options=['Yes','No'])
        Fluid = st.sidebar.selectbox('Fluid', options=['PR','HC','Propylene','Sulfuric acid','Methyl Methacrylate',
                                                       'Ammonia','Hexane','Acrylonitrile','Liquefied Petroleum Gas','CO','H','BD',
                                                       'OG','LPG','ET','VCM','Line to Flare','Treated Water','Nitrogen','TEA',
                                                       'Fire Water','EDC Vapor','P','Butene','CAR','MEG','Utility Air','SL','EDC',
                                                       'EG','BU' ])
        Operating_Temp_min = st.sidebar.slider('Operating Temperature (minimum)',-50, 100,0)
        Operating_Temp_max = st.sidebar.slider('Operating Temperature (maximum)',-50, 100,0)
        Material = st.sidebar.selectbox('Material', options=['Carbon Steel','Stainless Steel'])
        Coating = st.sidebar.selectbox('Coating', options=['Yes','No'])
        Primer = st.sidebar.selectbox('Primer', options=['Inorganic zinc Self-cure Solvent base','No','Silicone','Inorganic zinc',
                                                         'Silicone (Black)','Epoxy-Phenolic'])
        Mist = st.sidebar.selectbox('Mist', options=['High-built Polyamide Epoxy','No'])
        Intermediate = st.sidebar.selectbox('Intermediate', options=['High-build Polyamide Epoxy','No','Polyamide epoxy','Epoxy-Phenolic'])
        Finish = st.sidebar.selectbox('Finish', options=['Aliphatic Polyurethane Final coat','No','Polyamide epoxy','Silicone','Aliphatic Acrilic Poliurethane'])
        Anti_Corrosion = st.sidebar.selectbox('Anti-Corrosion', options=['Yes','No'])
        Environment = st.sidebar.selectbox('Environment', options=['Marine','Moderate Climate or Industrial Environments (Rainfall: about 500-1000 mm/year)','Coastal/Marine'])
        
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

data_train = pd.read_csv('./04.Dataset_for_deployment/20230911_Final_Clean.csv')
raw_data = data_train.drop(columns=['Finding No.','Severity'])


#st.table(input_df)
df = pd.concat([input_df,raw_data],axis=0,ignore_index=True)


# Selects only the first row (the user input data)
df = df[:] 
#st.table(df)


# Displays the user input features
st.subheader('1. Features for Simulation')
#st.image('picture/simulation.png', width=45)

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
#scaler = StandardScaler()
#df = scaler.fit_transform(df)
#st.write(df)

# Reads in saved regression model



load_clf1 = pickle.load(open('20230912_RFModel.pkl','rb'))
load_clf2 = pickle.load(open('20230912_DTModel.pkl','rb'))


# Apply model to make predictions
predict = pd.DataFrame(df).iloc[:lastrow]
#prediction = load_clf2.predict(predict)
if select == 'Random Forest':
    prediction = load_clf1.predict(predict)
elif select == 'Decision Tree':
    prediction = load_clf2.predict(predict)

#----------------------------------------------------------

st.subheader(':blue[2. CUPS Prediction]')
st.image('picture/predictive-chart.png', width=45)
#st.write([prediction])
st.write('Severity')
st.write(prediction)
#-----------------------------------------------------------



st.subheader(':blue[3. Severity chart]')
st.write("3.1 Bar chart of severity")
#line_fig = px.line(uploaded_file,x='Start_time', y='prediction', title='Line chart of eeverity')
#st.plotly_chart(line_fig)
st.line_chart(prediction)

#st.write("3.2 Bar chart of severity")
#st.bar_chart(prediction)


#----------------------------------------------------------------
#Save prediction file to csv

predic = pd.Series(prediction, name='Severity')

df_final = pd.concat([input_df.iloc[:,1:14], pd.Series(predic)], axis=1)

@st.cache_data
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
