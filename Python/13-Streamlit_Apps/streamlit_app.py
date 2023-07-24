import streamlit as st
import pandas as pd

st.write("""
# My first app
Hello *world!*
""")

data = pd.read_csv("17_Optimization/credit_data.csv")
st.bar_chart(data)

st.latex(r''' e^{i\pi} + 1 = 0 ''')

#st.download_button('On the dl', data)
mycolor = st.color_picker('Pick a color')

st.write(mycolor)

newdata = st.file_uploader('File uploader')

meh = pd.read_csv(newdata)

st.write(meh.describe())
