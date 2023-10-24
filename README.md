# project1
#My 1st work on streamlit
import numpy as np
import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

df=pd.read_csv('US_Regional_Sales_Data.csv')
df.head()
df['Unit Price']=df['Unit Price'].str.replace(',','').astype(float)
df['Unit Cost']=df['Unit Cost'].str.replace(',','').astype(float)
df['Profit']=(df['Order Quantity']*(((df['Unit Price'])-(df['Unit Price']*df['Discount Applied']))- df['Unit Cost']))
def profits(x):
    if x<0:
        return "No"
    else:
        return "Yes"
df['Profitable']= df['Profit'].apply(profits)
df['Profitable'].value_counts()
df.rename(columns={'Order Quantity':'Order_Quantity','Unit Price':'Unit_Price','Unit Cost':'Unit_Cost'}, inplace=True)
df['Order_Quantity']=df['Order_Quantity'].astype(float)
X = df[['Order_Quantity', 'Unit_Price', 'Unit_Cost']]
y = df['Profit']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


st.write("""
# Profitability Prediction App

This app predicts the **Profit or loss**!
""")
st.write('---')
st.sidebar.header('Specify Input Parameters')

def user_input_features():
    Order_Quantity = st.sidebar.slider('Order_Quantity', X.Order_Quantity.min(), X.Order_Quantity.max(), X.Order_Quantity.mean(),step=1.0)
    Unit_Price = st.sidebar.slider('Unit_Price', X.Unit_Price.min(), X.Unit_Price.max(), X.Unit_Price.mean(),step=2.0)
    Unit_Cost = st.sidebar.slider('Unit_Cost', X.Unit_Cost.min(), X.Unit_Cost.max(), X.Unit_Cost.mean(),step=2.0)
    data = {'Order_Quantity':Order_Quantity,
            'Unit_Price': Unit_Price,
            'Unit_Cost': Unit_Cost}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Main Panel

st.set_option('deprecation.showPyplotGlobalUse', False)
# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')

# Build Regression Model
model = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1)
model.fit(X_train, y_train)
# Apply Model to Make Prediction
prediction = model.predict(df)


st.header('Prediction of Profit/Loss')
st.write(prediction)
if prediction>0:
    st.subheader('This was a profitable transaction')
else:
    st.subheader('This was a loss transaction')
st.write('---')


explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

st.header('Feature Importance')
plt.title('Feature importance based on SHAP values')
shap.summary_plot(shap_values, X_train)
st.pyplot(bbox_inches='tight')
st.write('---')

plt.title('Feature importance based on SHAP values (Bar)')
shap.summary_plot(shap_values, X_train, plot_type="bar")
st.pyplot(bbox_inches='tight')
