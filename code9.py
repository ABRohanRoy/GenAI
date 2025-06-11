from langchain_core.prompts import PromptTemplate
from langchain.chat_models import AzureChatOpenAI
import os
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
load_dotenv()

openai_api_base=os.getenv("AZURE_OPENAI_API_BASE"),
openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),

llm_model = AzureChatOpenAI(
    deployment_name=deployment_name,
    openai_api_base = openai_api_version,
    openai_api_key=openai_api_key,
    azure_endpoint=openai_api_base,
    model_name="gpt-4o",
    temperature=0.7,
    top_p=0.9,
    max_tokens=100,
)
# max number of response
examples=[
    {"question":"What is the capital of France?", "answer": "The capita of france is Paris"},
    {"question":"What is the capital of Germany?", "answer": "The capita of Germany is Berlin"},
    {"question":"What is the capital of Italy?", "answer": "The capita of Italy is Rome"},
]

examples_prompt=PromptTemplate(
    input_variables=["question","answer"]
    template="Question: {question}\nAnswer: {answer}",
)

few_shot_prompt=FewShotPromptTemplate(
    examples=examples,
    examples_prompt=examples_prompt,
    input_variables=["question"],
    suffic="Answer:",
    prefix="You are a helpful assistant. Answer the following question.",
)

Formatted_prompt=few_shot_prompt.format(question="What is the capital of Spain?")
print(Formatted_prompt)
LLMChain=LLMChain(
    llm=llm_model,
    prompt=few_shot_prompt,
)
result =LLMChain.invoke({"question":"What is the capital of Spain"})
print(Formatted_prompt)
print(result['text'])