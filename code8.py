from langchain_core.prompts import PromptTemplate
from langchain.chat_models import AzureChatOpenAI
import os
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
load_dotenv()

llm_model = AzureChatOpenAI(
    openai_api_base=os.getenv("AZURE_OPENAI_API_BASE"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    model_name="gpt-4o",
    temperature=0.7,
)
prompt=ChatPromptTemplate.from_messages(
[
    SystemMessage(
        content="You are a Aviation Enthusiast and you can debrief about any aircraft" \
        "If you dont know then tell the user that you dont know the answer. " 
    ),
    HumanMessage(
        content="Tell me the fastest aircraft and its speed"
    ),
    AIMessage(content="{answer}"),
]
)
LLMChain=LLMChain(llm=llm_model,
                  prompt=prompt)
result =LLMChain.invoke({})
print(result)