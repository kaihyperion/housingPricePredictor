import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pydeck as pdk
import os
from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, households_ix = 3,4,5,6
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room


    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:,rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]



model = joblib.load('predictor_model.pkl')


def load_map_data():
    file_path = os.path.join('datasets','housing','housing.csv')
    data = pd.read_csv(file_path)
    precision=1
    data['lat'] = data['latitude'].round(precision)
    data['lon'] = data['longitude'].round(precision)
    # Aggregate data
    grouped_data = data.groupby(['lat', 'lon']).agg(
        median_house_value=('median_house_value', 'mean'),
        population=('population', 'sum'),
        lat=('lat', 'first'),
        lon=('lon', 'first'),
        count=('lat', 'size')  # Count entries per group for weighting if desired
    ).reset_index(drop=True)
    grouped_data['radius'] = grouped_data['population']*5
    return grouped_data

def user_input_features():
    with st.form("input_form"):
        st.write("Input additional features for prediction:")
        latitude = st.number_input('Latitude', value=38.52,format="%.4f")
        longitude = st.number_input('Longitude',value=-121.46, format="%.4f")
        housing_median_age = st.number_input('Housing Median Age', value=29)
        total_rooms = st.number_input('Total Rooms', min_value=0, value=3873, step=1)
        total_bedrooms = st.number_input('Total Bedrooms', min_value=0, value=797, step=1)
        population = st.number_input('Population', min_value=0, value=2237, step=10)
        households = st.number_input('Households', min_value=0, value=706, step=1)
        median_income = st.number_input('Median Income', min_value=0.0, value=2.1736, step=0.01, format="%.1f")
        ocean_proximity = st.selectbox('Ocean Proximity', ('NEAR BAY', 'INLAND', '<1H OCEAN', 'ISLAND', 'NEAR OCEAN'))
        submit_button = st.form_submit_button("Predict")
        
        data = {
            'longitude': [longitude],
            'latitude': [latitude],
            'housing_median_age': [housing_median_age],
            'total_rooms': [total_rooms],
            'total_bedrooms': [total_bedrooms],
            'population': [population],
            'households': [households],
            'median_income': [median_income],
            'ocean_proximity': [ocean_proximity]
        }
        
        if submit_button:
            features= pd.DataFrame(data)
            prediction = model.predict(features)
            st.write(f"Predicted Housing Price: ${prediction[0]}")




# Main Application
def main():
    st.title('California Housing Price Prediction')
    data = load_map_data()
    
    view_state = pdk.ViewState(
        latitude=data['lat'].mean(),
        longitude =data['lon'].mean(),
        zoom=4.8,
        pitch=30
    )
    layer = pdk.Layer(
        "ColumnLayer",
        data=data,
        get_position=['lon','lat'],
        get_elevation='median_house_value',
        elevation_scale=.7,
        radius=10000,
        get_fill_color="""
            [255 * ((median_house_value - min_median_house_value) / (max_median_house_value - min_median_house_value)), 
             0, 
             255 * (1 - (median_house_value - min_median_house_value) / (max_median_house_value - min_median_house_value)), 
             150]
        """.replace('min_median_house_value', str(data['median_house_value'].min())).replace('max_median_house_value', str(data['median_house_value'].max())),
        pickable=True,
        auto_highlight=True
        )
        
    map = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style='mapbox://styles/mapbox/light-v9',
        tooltip={"text": "Median House Value: {median_house_value}, Population: {population}"}
        )
    st.pydeck_chart(map)
    
    user_input_features()



if __name__ == "__main__":
    main()