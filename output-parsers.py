from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser

# Load .env file
load_dotenv()

# instantiate the model
llm = ChatOpenAI(model="gpt-4o-mini")

parser = CommaSeparatedListOutputParser()

#prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "you make a list of countries that starts with" ),
    ("human", "{input}")
])

# create LLM chain
llm_chain = prompt | llm | parser


response = llm_chain.invoke({"input": "G"})
print(response)