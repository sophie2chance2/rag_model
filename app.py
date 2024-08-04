import streamlit as st
import rag_model

# Title of the app
st.title("Model Query App")

# Department selection
department = st.selectbox(
    "Select your department:",
    ("Marketing", "Research", "Other")
)
# Additional text input for department if "Other" is selected
if department == "Other":
    department = st.text_input("Please specify your department:")

# Model selection
model = st.selectbox(
    "Select the model you would like to use:",
    ("Mistral", "Cohere")
)

# Question input
question = st.text_input("Enter the question you would like to ask the model:")

# Submit button
if st.button("Submit"):
    # Here you would add the code to query the selected model with the question
    # For now, we'll just display the inputs
    st.write(f"Department: {department}")
    st.write(f"Model: {model}")
    st.write(f"Question: {question}")
    # Example of querying a model (placeholder)
    response = rag_model.call_model(question, department.lower(), model.lower())
    # st.write(f"Response: {response}")

# Note: Replace 'query_model' with the actual function to query the model.
# This is a placeholder to demonstrate how to capture inputs and use them.