import streamlit as st
import pandas as pd
import pickle

# Load the pickled model
with open('model_arima.pkl', 'rb') as file:
    model = pickle.load(file)

# Define a function to make predictions
def predict_sales(date, sales_volume):
    # Create a dataframe with the input values
    data = pd.DataFrame({'Sales_volume': [sales_volume]})
    # Make a prediction using the loaded model
    try:
        prediction = model.forecast(steps=1, exog=data)[0]
        return prediction
    except Exception as e:
        st.error('Error: {}'.format(str(e)))


# Create a Streamlit app
st.title('Sales Volume Prediction')
# Create input fields for the user to enter data
date = st.date_input('Enter the date:')
sales_volume = st.number_input('Enter the sales volume:')
# Make a prediction when the user clicks the 'Predict' button
if st.button('Predict'):
    prediction = predict_sales(date, sales_volume)
    if prediction is not None:
        st.write('The predicted sales volume is:', round(prediction, 2))
