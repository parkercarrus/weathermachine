import pandas as pd
import numpy as np
import joblib, time, tensorflow as tf
from datetime import datetime
from silence_tensorflow import silence_tensorflow

silence_tensorflow()

PAST_WEATHER_PATH = 'data/past_weather.csv'
PATH_TO_WEATHER_WITH_CYCLICAL = 'data/past_weather_with_cyclical_features.csv'
PATH_TO_RF = 'models/rf1_temp.pkl'
PATH_TO_NN = 'models/temp2.keras'


def get_cyclical_datetime():
    dt = datetime.now()
    # Normalize time components
    seconds_in_a_day = 24 * 60 * 60
    seconds = dt.hour * 3600 + dt.minute * 60 + dt.second
    time_normalized = (seconds / seconds_in_a_day) * 2 * np.pi
    
    # Calculate days from midsummer (June 21st)
    midsummer = datetime(dt.year, 6, 21)
    days_from_midsummer = (dt - midsummer).days

    # Normalize days from midsummer to [0, 365)
    days_from_midsummer_normalized = (days_from_midsummer + 172) % 365
    day_of_year_normalized = (days_from_midsummer_normalized / 365) * 2 * np.pi

    # Calculate sin and cos for time
    sin_time = np.sin(time_normalized)
    cos_time = np.cos(time_normalized)

    # Calculate sin and cos for days from midsummer
    sin_day_of_year = np.sin(day_of_year_normalized)
    cos_day_of_year = np.cos(day_of_year_normalized)

    return {
        'time_sin': sin_time,
        'time_cos': cos_time,
        'year_day_sin': sin_day_of_year,
        'year_day_cos': cos_day_of_year
    }

class RandomForest:
    def __init__(self, data):
        self.data = data
        # load random forest model
        path_to_rf = PATH_TO_RF
        try:
            self.rf_model = joblib.load(path_to_rf)
        except FileNotFoundError:
            raise FileNotFoundError(f'Random Forest Classifier could not be accessed from {path_to_rf}')
        except Exception as e:
            raise Exception(f'An error occured while loading the Random Forest Classifier: {e}')

    def process_data(self, data):
        """
        Returns numpy array of shape (1,16)

        Parameters: data(dict): Dictionary containing current weather data
        """
        try:
            # extract weather features from JSON request to server
            temp = data.get('temp')
            humidity = data.get('humidity')
            pressure = data.get('pressure')

            if temp is None or humidity is None or pressure is None:
                raise ValueError('Missing required weather data')
            
            # get cyclical time features
            trig_dt = get_cyclical_datetime()
            time_sin = trig_dt.get('time_sin')
            time_cos = trig_dt.get('time_cos')
            year_day_sin = trig_dt.get('year_day_sin')
            year_day_cos = trig_dt.get('year_day_cos')

            # create a list of current weather features
            features_list = [time_sin, time_cos, year_day_sin, year_day_cos, temp, humidity, pressure] # removing precip for now... RFclassifier does not expect it. will need to sort out models later

            # get past weather features (3 hours)
            path_to_past_data = PAST_WEATHER_PATH
            past_weather_csv = pd.read_csv(path_to_past_data).tail(3)

            if past_weather_csv.shape[0] < 3:
                raise ValueError(f'Insufficient past weather data stored in {path_to_past_data}')

            # append past weather data to the feature list
            for i in range(3):
                for j in range(3):
                    features_list.append(past_weather_csv.values[2-i][j])

            # convert features list to a numpy array and reshape to desired input shape
            arr = np.array(features_list)
            arr = arr.reshape(1,16)     

            return arr

        except Exception as e:
            raise RuntimeError(f'Error processing data: {e}')
        
    def is_Raining(self) -> bool:
        """
        Returns a boolean value correspoding to the RF model's prediction of rain

        Parameters: None
        """
        try:
            inputs = self.process_data(self.data)
            eval = self.rf_model.predict(inputs)
            if eval == np.array([0]):
                return False
            elif eval == np.array([1]):
                return True
            else: 
                raise ValueError(f'Error with output of Random Forest Classifier: {eval}')
        except Exception as e:
            raise Exception(f'An error occured while evaluating the Random Forest Classifier: {e}')
        
from sklearn.preprocessing import MinMaxScaler

MAX_TEMP = 312.41
MIN_TEMP = 248.42

MAX_PRESSURE = 1047.0
MIN_PRESSURE = 984.0

MAX_HUMIDITY = 100
MIN_HUMIDITY = 7

def fahrenheit(k):
    return 1.8 * (k - 273.15) + 32

def create_sequences(X, sequence_length):
    Xs = []
    for i in range(len(X) - sequence_length):
        Xs.append(X[i:(i + sequence_length)])
    return np.array(Xs)

class LSTMModel:
    def __init__(self, data):
        self.current_weather = data
        self.past_weather = pd.read_csv(PATH_TO_WEATHER_WITH_CYCLICAL)
        self.next_hr_model = tf.keras.models.load_model(PATH_TO_NN)
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

    def process_data(self):
        now = datetime.now()
        current_cyclical_datetime = {
            'time_sin': np.sin(2 * np.pi * now.hour / 24),
            'time_cos': np.cos(2 * np.pi * now.hour / 24),
            'year_day_sin': np.sin(2 * np.pi * now.timetuple().tm_yday / 365.25),
            'year_day_cos': np.cos(2 * np.pi * now.timetuple().tm_yday / 365.25),
        }

        current_weather_df = pd.DataFrame([{
            **current_cyclical_datetime,
            'temp': self.current_weather['temp'],
            'pressure': self.current_weather['pressure'],
            'humidity': self.current_weather['humidity']
        }])

        weather_data = pd.concat([self.past_weather, current_weather_df], ignore_index=True)

        X = weather_data[['time_sin', 'time_cos', 'year_day_sin', 'year_day_cos', 'temp', 'pressure', 'humidity']]
        self.scaler_X.fit(X)
        self.scaler_y.fit(weather_data[['temp']])

        X_scaled = self.scaler_X.transform(X)
        X_seq = create_sequences(X=X_scaled, sequence_length=24)

        last_sequence = X_seq[-1]
        last_sequence = last_sequence.reshape(1, 24, X.shape[1])

        return last_sequence
    
    def get_next_hour(self):
        last_sequence = self.process_data()
        prediction_scaled = self.next_hr_model.predict(last_sequence)
        prediction = self.scaler_y.inverse_transform(prediction_scaled).flatten()
        return prediction
    
import requests 

def get_weather():
    api_key = '8182f964fba00e24c30bec624f35c05e'
    latitude = '41.5354699'
    longitude = '-81.421139'
    # Construct the URL for the API call
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={api_key}&units=standard"
    
    # Make the request to OpenWeatherMap API
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON data
        weather_data = response.json()

        t = weather_data['main'].get('temp')
        p = weather_data['main'].get('pressure')
        h = weather_data['main'].get('humidity')

        data = {
            'temp': t,
            'pressure': p,
            'humidity': h
        }

        return data
    else:
        return {"error": "Failed to fetch data", "status_code": response.status_code}

# functions to make simple predicitons

def make_predict():
    data = get_weather()
    classifier = RandomForest(data)
    regressor = LSTMModel(data)
    raining = classifier.is_Raining()
    temp = regressor.get_next_hour()

    print(f'next hour temperature will be {fahrenheit(temp)} and it will {"not" if not raining else ""} rain')
    print(f'current temp: {fahrenheit(data.get("temp"))}')


def make_predict_from_data(data):
    classifier = RandomForest(data)
    regressor = LSTMModel(data)
    raining = classifier.is_Raining()
    temp = regressor.get_next_hour()

    print(f'next hour temperature will be {fahrenheit(temp)} and it will {"not" if not raining else ""} rain')
    print(f'current temp: {fahrenheit(data.get("temp"))}')

def make_predict_sci(data):
    classifier = RandomForest(data)
    regressor = LSTMModel(data)
    raining = classifier.is_Raining()
    temp = regressor.get_next_hour()

    return fahrenheit(temp), raining

def get_all(data=None):
    if data is None:
        data = get_weather()
    # initialize models
    rf = RandomForest(data)
    nn = LSTMModel(data)

    

        # call function to get sequence here 



