from langchain_core.prompts import PromptTemplate
from langchain.chat_models import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os
load_dotenv()
from langchain.memory import ConversationBufferMemory

api_key=os.getenv("AZURE_OPENAI_API_KEY")
api_base=os.getenv("AZURE_OPENAI_ENDPOINT")
deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
api_version=os.getenv('AZURE_OPENAI_API_VERSION')

llm=AzureChatOpenAI(
    deployment_name=deployment_name,
    api_version=api_version,
    api_key=api_key,
    azure_endpoint=api_base,
    model_name="gpt-4o",
    temperature=0.9,
    top_p=0.9,
    max_tokens=100,
)
prompt = PromptTemplate(
    input_variables=["input_text"],
    template="You are a master of different domains. Answer all questions accordingly: {input_text}"
)
memory=ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
)
chain=LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
)

while True:
    input_text=input("Hey What can i do for you?(or type exit to stop runnign the code): ").strip()
    if input.lower()=="exit":
        break
    result =chain.invoke({"input_text": input_text})
    print(f"Response for {input}: {result['text']}")
    print("---------------------------------------------------------------------------------------")
    print(memory.buffer)

