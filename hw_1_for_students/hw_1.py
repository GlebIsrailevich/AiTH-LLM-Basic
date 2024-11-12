#pip install --upgrade --quiet gigachain==0.2.6 gigachain_community==0.2.6 gigachain-cli==0.0.25 duckduckgo-search==6.2.4 pyfiglet==1.0.2 langchain-anthropic llama_index==0.9.40 pypdf==4.0.1 sentence_transformers==2.3.1

import os
import getpass
import requests
import json

from langchain.chat_models.gigachat import GigaChat

from langchain.schema import HumanMessage, SystemMessage

from langchain_community.llms import HuggingFaceHub

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Optional, Union

from langchain.tools import tool
# from langchain.agents import AgentExecutor, create_gigachat_functions_agent

from langchain_community.tools import DuckDuckGoSearchRun

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
# from llama_index.llms.huggingface import HuggingFaceLLM
# from llama_index.prompts.prompts import SimpleInputPrompt
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.core import ServiceContext
from llama_index.core import Settings
# from creds import response
from dotenv import load_dotenv
from llama_index.embeddings.langchain import LangchainEmbedding
load_dotenv('.env')
sb_auth_data = os.getenv('SB_AUTH_DATA')

# # 1. GigaChat
# Define GigaChat throw langchain.chat_models

def get_giga(giga_key: str) -> GigaChat:
    giga_model = GigaChat(credentials=giga_key,
                model="GigaChat-Pro", timeout=30, max_tokens=8000, verify_ssl_certs=False)
    return giga_model

def test_giga():
    giga_key = getpass.getpass(sb_auth_data)
    # giga_key = sb_auth_data
    giga = get_giga(giga_key)
    assert giga != None



# # 2. Prompting
# ### 2.1 Define classic prompt

# Implement a function to build a classic prompt (with System and User parts)
def get_prompt(user_content: str)-> List[Union[SystemMessage, HumanMessage]]:
    messages = [
    SystemMessage(
        content="Ты лингвист приветствуй собеседника на разных языках"
    )
    ]
    messages.append(HumanMessage(content=user_content))
    
    return messages

# Let's check how it works
def tes_prompt():
    giga_key = getpass.getpass(sb_auth_data)
    # giga_key = sb_auth_data
    giga = get_giga(giga_key)
    user_content = 'Hello!'
    prompt = get_prompt(user_content)
    assert len(prompt) == 2
    res = giga(prompt)
    print(res.content)

# ### 3. Define few-shot prompting

# Implement a function to build a few-shot prompt to count even digits in the given number. The answer should be in the format 'Answer: The number {number} consist of {text} even digits.', for example 'Answer: The number 11223344 consist of four even digits.'
def get_prompt_few_shot(number: str) -> List[HumanMessage]:

    few_shot_prompt = """
Пример 1: Сколько чётных цифр в числе: 12?  
        - Первая цифра — 1, это не чётная цифра, количество чётных цифр пока 0.  
        - Вторая цифра — 2, она чётная, количество чётных цифр стало 1.  
        - Других цифр нет, количество чётных цифр — 1.  
        - Ответ: Число 12 состоит из 1 чётной цифры.

Пример 2: Сколько чётных цифр в числе: 22247?  
        - Первая цифра — 2, это чётная цифра, количество чётных цифр — 1.  
        - Вторая цифра — 2, это чётная цифра, количество чётных цифр 2.  
        - Третья цифра — 2, она чётная, количество чётных цифр стало 3.  
        - Четвертая цифра — 4, она чётная, количество чётных цифр стало 4
        - Пятая цифра — 7, это не чётная цифра, количество чётных цифр остаётся 4.  
        - Других цифр нет, количество чётных цифр — 4.  
        - Ответ: Число 22247 состоит из 4 чётных цифр.

Пример 3: Сколько чётных цифр в числе: 1234567890?  
        - Первая цифра — 1, это не чётная цифра, количество чётных цифр пока 0.  
        - Вторая цифра — 2, это чётная цифра, количество чётных цифр — 1.  
        - Следующая цифра — 3, это не чётная цифра, количество чётных цифр всё ещё 1.  
        - Третья цифра — 4, это чётная цифра, количество чётных цифр стало 2.  
        - Четвёртая цифра — 5, это не чётная цифра, количество чётных цифр всё ещё 2.  
        - Пятая цифра — 6, это чётная цифра, количество чётных цифр стало 3.  
        - Шестая цифра — 7, это не чётная цифра, количество чётных цифр остаётся 3.  
        - Седьмая цифра — 8, это чётная цифра, количество чётных цифр стало 4.  
        - Восьмая цифра — 9, это не чётная цифра, количество чётных цифр всё ещё 4.  
        - Девятая цифра — 0, это чётная цифра, количество чётных цифр стало 5.  
        - Других цифр нет, количество чётных цифр — 5.  
        - Ответ: Число 1234567890 состоит из 5 чётных цифр.

Пример 4: Сколько четных цифр в числе: 936244188:
        - Первая цифра — 9, это не чётная цифра, количество чётных цифр пока 0.  
        - Вторая цифра — 3, это не чётная цифра, количество чётных цифр — 0.  
        - Следующая цифра — 6, это чётная цифра, количество чётных цифр становится 1.  
        - Третья цифра — 2, это чётная цифра, количество чётных цифр стало 2.  
        - Четвёртая цифра — 4, это чётная цифра, количество чётных цифр всё ещё 3.  
        - Пятая цифра — 4, это чётная цифра, количество чётных цифр становится 4.  
        - Шестая цифра — 1, это не чётная цифра, количество чётных цифр все еще 4.
        - Седьмая цифра — 8, это чётная цифра, количество чётных цифр становится 5.
        - Восьмая цифра — 8, это чётная цифра, количество чётных цифр становится 6.
        - Других цифр нет, количество чётных цифр — 6.  
        - Ответ: Число 936244188 состоит из 6 чётных цифр.

        Сколько чётных цифр в числе: {number}?  
        """
    return few_shot_prompt

# Let's check how it works
def test_few_shot():
    giga_key = getpass.getpass(sb_auth_data)
    # giga_key = sb_auth_data
    giga = get_giga(giga_key)
    number = '62388712774'
    prompt = get_prompt_few_shot(number)
    prmt = PromptTemplate(template=prompt, input_variables=['number'])
    chain = prmt | giga 
    res = chain.invoke({'number': number})

    answer = res.content[res.content.rfind('Answer:'):]
    assert answer == 'Answer: The number 62388712774 consist of 6 even digits.'
    print(res.content)

# # 4. Llama_index
# Implement your own class to use llama_index. You need to implement some code to build llama_index across your own documents. For this task you should use GigaChat Pro.
class LlamaIndex:
    def __init__(self, path_to_data: str, llm: GigaChat):
        self.system_prompt="""
        You are a Q&A assistant. Your goal is to answer questions as
        accurately as possible based on the instructions and context provided.
        """
        self.documents=SimpleDirectoryReader(path_to_data).load_data()
        self.embed_model=LangchainEmbedding(
            HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))
                
        Settings.llm = llm
        Settings.chunk_size = 1024
        Settings.embed_model = self.embed_model

        index=VectorStoreIndex.from_documents(self.documents)
        self.query_engine=index.as_query_engine()

        
    def query(self, user_prompt: str) -> str:
        user_input = self.system_prompt + user_prompt
        response = self.query_engine.query(user_input)

        return response


# Let's check
def test_llama_index():
    giga_key = getpass.getpass(sb_auth_data)
    # giga_key = sb_auth_data
    giga_pro = GigaChat(credentials=giga_key, model="GigaChat-Pro", timeout=30, verify_ssl_certs=False)

    llama_index = LlamaIndex("data/", giga_pro)
    res = llama_index.query('what is attention is all you need?')
    assert res != ''
    print(res)

    

