from langchain_openai import AzureChatOpenAI  # instead of langchain
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()

llm = AzureChatOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

response = llm.invoke([HumanMessage(content="Where is Marwadi University located?")])
print(response.content)
