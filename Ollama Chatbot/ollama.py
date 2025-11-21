import os
from dotenv import load_dotenv  
# to load the env file
from langchain_community.llms import Ollama  
# for loading the ollama model
import streamlit as st  
# for making the streamlit work
from langchain_core.prompts import ChatPromptTemplate  
# for making the prompt ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser 
 # for getting the proper str output (clean output)

load_dotenv()  # to load the env file

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the question asked"),
        ("user", "Question:{question}")
    ]
)

def generate_response(question, llm, temperature, max_tokens):
    llm = Ollama(model=llm, temperature=temperature, num_predict=max_tokens)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({"question": question})
    return answer

st.title("Enhanced Q&a Bot With OpenAI")
llm = st.sidebar.selectbox("Select open source model", ["mistral", "llama3.2:1b"])

temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

st.write("Ask me a question..")
user_input = st.text_input("You: ")

if user_input:
    # means input is provided by the user and we have to pass it to model to get the answerr
    answer_of_user_input = generate_response(user_input, llm, temperature, max_tokens)
    st.write(answer_of_user_input)
else:
    # means the user input is not provided
    st.write("Please provide the user input ")
