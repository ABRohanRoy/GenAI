from langchain_core.prompts import PromptTemplate
from langchain.chat_models import AzureChatOpenAI
import os
from langchain.chains import LLMChain
from dotenv import load_dotenv
load_dotenv()

# 
llm_model =AzureChatOpenAI(
    openai_api_base=os.getenv("AZURE_OPENAI_API_BASE"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    model_name="gpt-4o",
    temperature=0.7,
)

my_prompt= PromptTemplate.from_template(
    "Tell me about cricket in a concise manner"
)

# connect
result =LLMChain(llm=llm_model,
                 prompt=my_prompt)
result=result.invoke({})
print(result)