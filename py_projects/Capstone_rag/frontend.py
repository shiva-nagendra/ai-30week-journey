import streamlit as st
import requests

st.title("RAG AI Assistant")

query = st.text_input("Ask a question")

if st.button("Submit"):

    response = requests.post(
        "http://backend:8000/predict",
        json={"query": query}
    )

    data = response.json()

    st.write(data["answer"])