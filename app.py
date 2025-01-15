from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

# Load .env file
load_dotenv()

# instantiate the model
llm = ChatOpenAI(model="gpt-4o-mini")

#prompt template
prompt = ChatPromptTemplate([
    ("system", "You make a list of countries that starts with" ),
    ("human", "{input}")
])

parser = StrOutputParser()

# create LLM chain
llm_chain = prompt | llm | parser

# streamlit app
st.title("Langchain Example")
input_text = st.text_input("Enter a letter")

if input_text:
    response = llm_chain.invoke({"input": input_text})
    st.write(response)

