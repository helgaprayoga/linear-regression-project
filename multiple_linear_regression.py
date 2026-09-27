# Core Packages

import streamlit as st
import sklearn
import joblib
import os
import numpy as np


# Loading Model

def load_prediction_model(model_file):
	loaded_model = joblib.load(
    	open(os.path.join(model_file), "rb")
	)

	return loaded_model


def main():
    
	"""Multiple Linear Regression — IHSG"""

	st.title("Prediksi Harga Penutupan IHSG")
	
	html_templ = """
    <div style="background-color:#726E6D;padding:10px;">
        <h3 style="color:white">
            Prediksi Harga Penutupan IHSG Menggunakan
            Multiple Linear Regression
        </h3>
    </div>
    """
	
	st.markdown(
        html_templ,
        unsafe_allow_html=True
    )
	
	activity = [
        "Prediksi Harga Penutupan IHSG",
        "Apa itu Multiple Linear Regression?"
    ]
	
	choice = st.sidebar.selectbox(
        "Menu",
        activity
    )
	
	# Prediksi IHSG
	if choice == "Prediksi Harga Penutupan IHSG":
		
		st.subheader("Prediksi Harga Penutupan IHSG")
		
		st.write(
            "Masukkan data IHSG yang ingin diprediksi."
        )
		
		# Input Open
		open_price = st.number_input(
            "Harga Pembukaan (Open)",
            min_value=0.0,
            value=6315.53,
            step=0.01
        )
		
		# Input High
		high_price = st.number_input(
            "Harga Tertinggi (High)",
            min_value=0.0,
            value=6328.41,
            step=0.01
        )
		
		# Input Low
		low_price = st.number_input(
            "Harga Terendah (Low)",
            min_value=0.0,
            value=6231.83,
            step=0.01
        )
		
		# Input Volume
		volume = st.number_input(
            "Volume Perdagangan",
            min_value=0.0,
            value=23850000000.0,
            step=100000000.0
        )
		
		if st.button("Prediksi"):
			# Load trained model
			regressor = load_prediction_model(
				"models/multiple_linear_regression_ihsg.pkl"
			)

			# Prepare input data
			new_data = np.array([
				[
					open_price,
					high_price,
					low_price,
					volume
				]
			])

			# Make prediction
			predicted_close = regressor.predict(
				new_data
			)

			# Display prediction
			st.info(
				"Prediksi Harga Penutupan IHSG: {}".format(
					round(predicted_close[0][0], 2)
				)
			)

	# Penjelasan Multiple Linear Regression
	elif choice == "Apa itu Multiple Linear Regression?":
		st.subheader(
			"Apa itu Multiple Linear Regression?"
		)

		st.write(
			"""
			Multiple Linear Regression atau Regresi Linier Berganda
			adalah metode statistik yang digunakan untuk memprediksi
			suatu variabel berdasarkan dua atau lebih variabel bebas.
			"""
		)

		st.write(
			"Pada model IHSG ini digunakan:"
		)

		st.write(
			"""
			Variabel Independen (X):

			- Open (Harga Pembukaan)
			- High (Harga Tertinggi)
			- Low (Harga Terendah)
			- Volume (Volume Perdagangan)

			Variabel Dependen (Y):

			- Close (Harga Penutupan)
			"""
		)

		st.write(
			"""
			Persamaan regresi yang digunakan:

			Close = b + β1(Open) + β2(High) + β3(Low) + β4(Volume)
			"""
		)

if __name__ == "__main__":
	main()