import streamlit as st
import requests

st.title("Streamlit App")

def trigger_action(data):
    st.write(f"Triggered action with data: {data}")

if st.button("Trigger Action"):
    trigger_action("Data from Streamlit button")

@st.experimental_singleton
def listen_for_trigger():
    if st.experimental_get_query_params().get("trigger_action"):
        data = st.experimental_get_query_params()["trigger_action"][0]
        trigger_action(data)

listen_for_trigger()
