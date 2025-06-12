from langchain_core.prompts import PromptTemplate
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os
load_dotenv()
from langchain.memory import ConversationSummaryBufferMemory

api_key=os.getenv("AZURE_OPENAI_API_KEY")
api_base=os.getenv("AZURE_OPENAI_ENDPOINT")
deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
api_version=os.getenv('AZURE_OPENAI_API_VERSION')

llm=AzureChatOpenAI(
    deployment_name=deployment_name,
    api_version=api_version,
    api_key=api_key,
    azure_endpoint=api_base,
    model_name="gpt-4",
    temperature=0.9,
    top_p=0.9,
    max_tokens=100,
)
prompt=PromptTemplate.from_template(
    """You are a helpfil assistant.
    chat_history: {chat_history}
    Question: {question}"""   
)
memory=ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=1000,
    return_messages=True,
    input_key="question",
    output_key="text",
    memory_key="chat_history",
)
chain=LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
)
while True:
    user_input=input("You: ")
    if user_input.lower() =="exit":
        break
    result =chain.invoke({"question":user_input})
    print(f"Assistant: {result['text']}")
    print("-----------------------------------------------------")
    print(memory.buffer)