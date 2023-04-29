# źródło danych [https://www.kaggle.com/c/titanic/](https://www.kaggle.com/c/titanic)

import streamlit as st
import pickle
from datetime import datetime
startTime = datetime.now()
# import znanych nam bibliotek

import pathlib
from pathlib import Path

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

filename = "model.sv"
model = pickle.load(open(filename,'rb'))
# otwieramy wcześniej wytrenowany model

# sex_d = {0:"Kobieta",1:"Mężczyzna"}
# pclass_d = {0:"Pierwsza",1:"Druga", 2:"Trzecia"}
# embarked_d = {0:"Cherbourg", 1:"Queenstown", 2:"Southampton"}
# o ile wcześniej kodowaliśmy nasze zmienne, to teraz wprowadzamy etykiety z ich nazewnictwem

def main():

	st.set_page_config(page_title="Czy zdrowy")
	overview = st.container()
	left, right = st.columns(2)
	prediction = st.container()

	st.image("https://pliki.well.pl/i/01/51/77/015177_r2_1320.jpg")

	with overview:
		st.title("Czy zdrowy?")

	# with left:
	# 	sex_radio = st.radio( "Płeć", list(sex_d.keys()), format_func=lambda x : sex_d[x] )
	# 	pclass_radio = st.radio("Klasa", list(pclass_d.keys()), format_func=lambda x : pclass_d[x])
	# 	embarked_radio = st.radio( "Port zaokrętowania", list(embarked_d.keys()), index=2, format_func= lambda x: embarked_d[x] )

	with right:
		symptoms_slider = st.slider("Objawy", min_value=1, max_value=5)
		age_slider = st.slider("Wiek", value=1, min_value=1, max_value=90)
		diseases_slider = st.slider("Choroby", min_value=0, max_value=5)
		height_slider = st.slider("Wzrost", min_value=167, max_value=200)
		drugs_slider = st.slider("Leki", min_value=1, max_value=4)

	data = [[symptoms_slider, age_slider, diseases_slider, height_slider, drugs_slider]]
	healthy = model.predict(data)
	s_confidence = model.predict_proba(data)

	with prediction:
		st.subheader("Czy taka osoba jest zdrowa?")
		st.subheader(("Tak" if healthy[0] == 1 else "Nie"))
		st.write("Pewność predykcji {0:.2f} %".format(s_confidence[0][healthy][0] * 100))

if __name__ == "__main__":
    main()
