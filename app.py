# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor

# Load dataset
#data = pd.read_csv("/Users/m.farhanrais/Documents/GitHub/DSI-SG-39/My Projects/Project 2/data/train_cleaned2.csv")
# For Streamlit:
data = pd.read_csv("train_cleaned2.csv")

# Define the features (X) and target variable (y)
X = data[['floor_area_sqm',
        'tranc_year',
        'mid_storey',
        'hdb_age',
        'max_floor_lvl',
        'year_completed',
        'total_dwelling_units',
        '4room_sold',
        '5room_sold',
        'exec_sold',
        'latitude',
        'longitude',
        'mall_nearest_distance',
        'mall_within_2km',
        'hawker_within_2km',
        'hawker_nearest_distance',
        'hawker_food_stalls',
        'hawker_market_stalls',
        'mrt_nearest_distance',
        'bus_stop_nearest_distance',
        'pri_sch_nearest_distance',
        'vacancy',
        'sec_sch_nearest_dist',
        'cutoff_point',
        'town_proxy']]
y = data['resale_price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Create and train a linear regression model
#model = LinearRegression()
#model.fit(X_train, y_train)

# Model v2
model = KNeighborsRegressor(n_neighbors=3)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics (optional)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Streamlit UI
st.title('HDB Resale Price Predictor')

# Sidebar with user input
st.sidebar.header('User Input')
floor_area_sqm = st.sidebar.number_input('1. Floor Area (sqm)', step =1, value=100, min_value=31, max_value=280)
tranc_year = st.sidebar.number_input('2. Year of resale transaction', step =1, value=2015, min_value=2012)
mid_storey = st.sidebar.number_input('3. Median value of storey_range', step =1, value=14, min_value=2, max_value=50)
hdb_age = st.sidebar.number_input('4. Number of years from lease_commence_date to present year', step =1, value=23, min_value=2, max_value=55)
max_floor_lvl = st.sidebar.number_input('5. Highest floor of the resale flat', step =1,value=18, min_value=2, max_value=50)
year_completed = st.sidebar.number_input('6. Year which construction was completed for resale flat', step =1, value=1996, min_value=1949, max_value=2018)
total_dwelling_units = st.sidebar.number_input('7. Total number of residential dwelling units in the resale flat', step =1, value=128, min_value=2, max_value=570)
four_room_sold = st.sidebar.number_input('8. Number of 4-room residential units in the resale flat', step =1, value=56, min_value=0, max_value=316)
five_room_sold = st.sidebar.number_input('9. Number of 5-room residential units in the resale flat', step =1, value=64, min_value=0, max_value=164)
exec_sold = st.sidebar.number_input('10. Number of executive type residential units in the resale flat block', step =1, value=8, min_value=0, max_value=135)
latitude = st.sidebar.number_input('11. Latitude based on postal code', step =0.000000001, format="%.9f", value=1.398304547)
longitude = st.sidebar.number_input('12. Longitude based on postal code', step =0.0000001, format="%.7f", value=103.7498178)
mall_nearest_distance = st.sidebar.number_input('13. Distance (metres) to the nearest mall', step =1, value=305, min_value=1)
mall_within_2km = st.sidebar.number_input('14. Number of malls within 2km', step =1, value=4, min_value=0)
hawker_within_2km = st.sidebar.number_input('15. Number of hawker centres within 2km', step =1, value=0, min_value=0)
hawker_nearest_distance = st.sidebar.number_input('16. Distance (metres) to the nearest hawker centre', step =1, value=3352, min_value=1)
hawker_food_stalls = st.sidebar.number_input('17. Number of hawker food stalls in the nearest hawker centre', step =1, value=28, min_value=0)
hawker_market_stalls = st.sidebar.number_input('18. Number of hawker and market stalls in the nearest hawker centre', step =1, value=45, min_value=0)
mrt_nearest_distance = st.sidebar.number_input('19. Distance (metres) to the nearest MRT station', step =1, value=282, min_value=1)
bus_stop_nearest_distance = st.sidebar.number_input('20. Distance (metres) to the nearest bus stop', step =1, value=171, min_value=1)
pri_sch_nearest_distance = st.sidebar.number_input('21. Distance (metres) to the nearest primary school', step =1, value=214, min_value=44)
vacancy = st.sidebar.number_input('22. Number of vacancies in the nearest primary school', step =1, value=55, min_value=0)
sec_sch_nearest_dist = st.sidebar.number_input('23. Distance (metres) to the nearest secondary school', step =1, value=446, min_value=36)
cutoff_point = st.sidebar.number_input('24. PSLE cutoff point of the nearest secondary school', step =1, value=260, min_value=212, max_value=260)
town_proxy = st.sidebar.number_input('25. GCS distance to central Singapore', step =0.0000000000000000001, format="%.19f", value=0.022136445, min_value=0.0000466134911083045, max_value=0.031179823)
prediction_button = st.sidebar.button('Predict')

# Prediction logic
if prediction_button:
    # Create a DataFrame with user input
    user_input = pd.DataFrame([[floor_area_sqm,
        tranc_year,
        mid_storey,
        hdb_age,
        max_floor_lvl,
        year_completed,
        total_dwelling_units,
        four_room_sold,
        five_room_sold,
        exec_sold,
        latitude,
        longitude,
        mall_nearest_distance,
        mall_within_2km,
        hawker_within_2km,
        hawker_nearest_distance,
        hawker_food_stalls,
        hawker_market_stalls,
        mrt_nearest_distance,
        bus_stop_nearest_distance,
        pri_sch_nearest_distance,
        vacancy,
        sec_sch_nearest_dist,
        cutoff_point,
        town_proxy]],
                              columns=['floor_area_sqm',
        'tranc_year',
        'mid_storey',
        'hdb_age',
        'max_floor_lvl',
        'year_completed',
        'total_dwelling_units',
        '4room_sold',
        '5room_sold',
        'exec_sold',
        'latitude',
        'longitude',
        'mall_nearest_distance',
        'mall_within_2km',
        'hawker_within_2km',
        'hawker_nearest_distance',
        'hawker_food_stalls',
        'hawker_market_stalls',
        'mrt_nearest_distance',
        'bus_stop_nearest_distance',
        'pri_sch_nearest_distance',
        'vacancy',
        'sec_sch_nearest_dist',
        'cutoff_point',
        'town_proxy'])
    
    # Make a prediction
    prediction = model.predict(user_input)
    
    # Change font size
    st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Display the prediction
    st.write(f'<p class="big-font">Predicted Resale Price: ${prediction[0]:,.2f}</p>', unsafe_allow_html=True)
    st.subheader('Model Performance')
    st.write(f'Mean Squared Error (MSE): {mse:.2f}')
    st.write(f'R-squared (R2): {r2:.2f}')
