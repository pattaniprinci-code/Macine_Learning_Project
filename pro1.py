import numpy as np 
import pandas as pd 
import  streamlit as st
import joblib 

model=joblib.load("logistic_model.pkl")
scaler=joblib.load("scaler.pkl")
required_col=joblib.load("columns.pkl")




st.header('Heart Dieases Prediction Model')
st.markdown("please fill the details for predict heart dieases")

# Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS',
    #    'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope',
    #    'HeartDisease'
    
age=st.slider('age' , 0 , 100 , 20)
Sex=st.selectbox("Sex" , ['Male' , 'Female'])
ChestPainType=st.selectbox("ChestPainType" , ['ATA', 'NAP', 'ASY', 'TA'])
RestingBP=st.selectbox("RestingBP" , [0 , 1])
Cholesterol=st.slider("Cholesterol" ,0 , 200 )
FastingBS=st.selectbox("FastingBS" ,[ 0 , 1])
RestingECG=st.selectbox("RestingECG" , ['Normal', 'ST', 'LVH'])
MaxHR=st.slider("MaxHR" , 0 , 250 , 10)
ExerciseAngina=st.selectbox("ExerciseAngina" , ['No' , 'Yes'])
Oldpeak=st.slider("Oldpeak" , -10 , 10)
ST_Slope=st.selectbox("ST_Slope" , ['Up', 'Flat', 'Down'])


if st.button("Predict"):
    input_data={
       "Age":age , 
       "Sex_"+Sex : 1 , 
       "ChestPainType_"+ChestPainType : 1,
       "RestingBP":RestingBP , 
       "Cholesterol":Cholesterol , 
       "FastingBS":FastingBS , 
       "RestingECG_"+RestingECG :1, 
       "MaxHR":MaxHR , 
       "ExerciseAngina_"+ExerciseAngina : 1, 
       "Oldpeak":Oldpeak , 
       "ST_Slope_"+ST_Slope :1, 
       
    }

    df=pd.DataFrame([input_data])
    
    for col in required_col:
        if col not in df.columns:
            df[col]=0
            
    df=df[required_col]

    X_scale=scaler.transform(df)
     
    
    pred=model.predict(X_scale)
    
    if (pred[0]==0):
        st.success('Low Chance Of Heart Dieases')
    
    else:
        st.warning('High Chance Of Heart Dieases')
        

    