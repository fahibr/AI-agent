import os
from dotenv import load_dotenv

# Conceptual LangChain/LangGraph setup
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

# Load environment variables from .env file
load_dotenv()


# Verify API key exists
if not os.getenv("AZURE_OPENAI_API_KEY"):
    raise ValueError("AZURE_OPENAI_API_KEY not found in .env file")


# Define a simple tool for the agent
@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


# Initialize the model and agent
model = ChatOpenAI(model="gpt-5-nano", api_key=os.getenv("AZURE_OPENAI_API_KEY"))
tools = [add_numbers]
agent = create_react_agent(model, tools=tools)

# Invoke the agent
response = agent.invoke({"messages": [("human", "What is 10 + 20?")]})

print("Agent Response:", response)