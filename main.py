import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool

load_dotenv()

if not os.getenv("AZURE_OPENAI_API_KEY"):
    raise ValueError("AZURE_OPENAI_API_KEY not found in .env file")

@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

model = AzureChatOpenAI(
    azure_deployment=os.getenv("DEPLOYMENT_NAME"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("API_VERSION", "2025-01-01-preview"),
    azure_endpoint=os.getenv(
        "AZURE_OPENAI_ENDPOINT",
        "https://aa-openai-sweden.openai.azure.com/",
    ),
    openai_api_version=os.getenv("OPENAI_API_VERSION", "2025-08-07")
)

tools = [add_numbers]
agent = create_agent(model, tools=tools)

response = agent.invoke({"messages": [("human", "What is 10 + 20?")]})
print("Agent Response:", response["messages"][1].content) #Print the last message from the agent's response