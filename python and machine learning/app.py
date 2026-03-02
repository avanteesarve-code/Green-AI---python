import streamlit as st
import pandas as pd
st.title('Welcome')
st.write("This is a simple Streamlit app.")
data = pd.read_csv(r"C:\Users\Avantee Sarve\OneDrive\Desktop\college\Green AI\appliance_energy.csv")
st.write(data)
st.line_chart(data)