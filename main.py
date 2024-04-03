import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load the trained model
@st.cache_resource
def load_model():
    rand = RandomForestClassifier(random_state=42)
    rand.fit(X_train, y_train)
    return rand

@st.cache_resource
def load_data():
    return pd.read_csv("seattle-weather.csv")

def get_user_input(data_columns):
    input_data = []
    for column in data_columns:
        value = st.number_input(f"Enter {column}:", step=0.01)
        input_data.append(value)
    return input_data

def generate_output(input_data):
    prediction = model.predict([input_data])
    decoded_prediction = encode_weather.inverse_transform(prediction)[0]
    return decoded_prediction

def main():
    st.title("Weather Prediction App")
    st.write("This app predicts the weather based on user input.")
  
    model = load_model()
    data = load_data()

    data_columns = data.columns[:-1]

    user_input = get_user_input(data_columns)

    if st.button("Predict Weather"):

        output = generate_output(user_input)

        st.write("Predicted Weather:", output)

if __name__ == "__main__":
    data = pd.read_csv("seattle-weather.csv")

    encode_weather = LabelEncoder()
    data['weather'] = encode_weather.fit_transform(data['weather'])

    X_train = data.drop(columns=['weather'])
    y_train = data['weather']

    model = load_model()

    main()
