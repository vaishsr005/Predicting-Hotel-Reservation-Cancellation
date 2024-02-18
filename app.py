import streamlit as st
import pandas as pd
import pickle
from PIL import Image

# Load the trained Random Forest model from the pickle file
with open('/Users/abhishekrai/Library/CloudStorage/GoogleDrive-niharikabatra111@gmail.com/My Drive/ML Projects/Predicting_Hotel_Reservation_Cancellation/random_forest_model.pkl', 'rb') as file:
    rf_model = pickle.load(file)

# Function to make predictions using the loaded model
def predict_booking_status(input_data):
    # Preprocess the input data (make sure it matches the format used for training)
    # Assuming 'input_data' is a DataFrame with features similar to the training data

    # Use the loaded model to make predictions
    predictions = rf_model.predict(input_data)
    
    prediction_prob = rf_model.predict_proba(input_data)
    
    return predictions, prediction_prob

# Streamlit App
def main():

    # Set page config to center the content
    st.set_page_config(layout="centered")

    #  # Add an image of hotel reservation
    # image = Image.open("/Users/abhishekrai/Library/CloudStorage/GoogleDrive-abhishek.rai085@gmail.com/My Drive/Niharika docs/UTA 2023 /DATA MINING/Datasets/Reservation_hotel_booking_project/Hotel.png")
    # st.image(image, use_column_width=True)

     # Add the heading with blue color
    st.markdown("<h1 style='text-align: center; color: maroon;'>Hotel Reservation Prediction App</h1>", unsafe_allow_html=True)

    st.write("Enter the following features to predict booking status:")
    # Create two columns for input fields
    col1, col2, col3 = st.columns(3)

    # st.title("Hotel Reservation Prediction App")
    
    with col1:
    # Input fields for the specified features with their respective ranges
        no_of_adults = st.selectbox("Number of Adults", [0, 1, 2, 3, 4], index=2)
        no_of_weekend_nights = st.selectbox("Number of Weekend Nights", [0, 1, 2, 3, 4, 5], index=1)
        no_of_week_nights = st.selectbox("Number of Week Nights", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], index=2)
        type_of_meal_plan = st.selectbox("Type of Meal Plan", [0, 1, 2], index=0)
        required_car_parking_space = st.selectbox("Required Car Parking Space", [0, 1], index=0)
        repeated_guest = st.selectbox("Repeated Guest", [0, 1], index=0)
        lead_time = st.slider("Lead Time", 0, 500, 50)

    with col3:
        # arrival_month = st.selectbox("Arrival Month", list(range(1, 13)), index=0)
        no_of_children = st.selectbox("Number of Children", [0, 1, 2, 3], index=2)
        room_type_reserved = st.selectbox("Room Type Reserved", [0, 1, 4, 3, 2, 5, 6] , index = 0)
        # market_segment_type = st.selectbox("Market Segment Type", [0, 1, 2, 3, 4], index=0)
        arrival_year = st.selectbox("Arrival Year", [2017, 2018], index=0)
        no_of_previous_cancellations = st.selectbox("Number of Previous Cancellations", [0, 1, 2, 3, 4, 5, 6, 11, 13], index=0)
        no_of_previous_bookings_not_canceled = st.selectbox("Number of Previous Bookings Not Canceled", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 31, 33, 34, 35, 36, 37, 44, 45, 46, 47, 48, 50, 51, 58], index=0)
        avg_price_per_room = st.slider("Average Price per Room", 0.0, 200.0, 100.0)
        no_of_special_requests = st.selectbox("Number of Special Requests", [0, 1, 2, 3, 4, 5], index=2)

    # Generate a DataFrame from user inputs
    input_data = pd.DataFrame({
        'no_of_adults': [no_of_adults],
        'no_of_weekend_nights': [no_of_weekend_nights],
        'no_of_week_nights': [no_of_week_nights],
        'type_of_meal_plan': [type_of_meal_plan],
        'required_car_parking_space': [required_car_parking_space],
        'lead_time': [lead_time],
        'arrival_year': [arrival_year],
        'no_of_children': [no_of_children],
        'room_type_reserved': [room_type_reserved],
        'repeated_guest': [repeated_guest],
        'no_of_previous_cancellations': [no_of_previous_cancellations],
        'no_of_previous_bookings_not_canceled': [no_of_previous_bookings_not_canceled],
        'avg_price_per_room': [avg_price_per_room],
        'no_of_special_requests': [no_of_special_requests]
    })
 

    with col2:
        # Padding to create space between input fields and button
        st.markdown("<p style='text-align: center; padding-top: 600px;padding-left:50px; padding-right:50px'> </p>", unsafe_allow_html=True)

        btn_predict = st.button("Predict", key="predict_button", help="Click to predict booking status")


    if btn_predict:
    # Make predictions using the loaded model
    # if st.button("Predict"):
        # input_data.to_csv('input_data.csv')
        predictions, prediction_prob = predict_booking_status(input_data)
       

        if predictions[0] == 0:

            st.markdown(f"<p style='font-size: 24px; color: green; text-align: center;'>Predicted Booking Status: Not Cancelled with {prediction_prob[0][0]*100:.2f}% probability.</p>", unsafe_allow_html=True)
        else:
            st.markdown(f"<p style='font-size: 24px; color: red; text-align: center;'>Predicted Booking Status: Cancelled with {prediction_prob[0][1]*100:.2f}% probability.</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
