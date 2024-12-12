import json
import boto3
import os
import time
import datetime
import PyPDF2
import csv
import re
import traceback
import requests
import base64
import operator
import uuid
import functools
import aiohttp

from botocore.config import Config
from botocore.exceptions import ClientError
from io import BytesIO
from urllib import parse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_aws import ChatBedrock
from langchain_community.vectorstores.opensearch_vector_search import OpenSearchVectorSearch
from multiprocessing import Process, Pipe
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores.faiss import FAISS

# from langchain.agents import tool
from langchain_core.tools import tool
from bs4 import BeautifulSoup
from pytz import timezone
from langchain_community.tools.tavily_search import TavilySearchResults
from PIL import Image
from opensearchpy import OpenSearch

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.prebuilt.tool_executor import ToolExecutor
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition
from pydantic.v1 import BaseModel, Field
from typing import Any, List, Tuple, Dict, Optional, cast, Literal, Sequence, Union
from typing_extensions import Annotated, TypedDict
from langchain_aws import AmazonKnowledgeBasesRetriever
from tavily import TavilyClient  

from dataclasses import dataclass, field
from langchain_core.runnables import RunnableConfig, ensure_config
from langchain_core.tools import InjectedToolArg
from langgraph.prebuilt import InjectedState
     
s3 = boto3.client('s3')
s3_bucket = os.environ.get('s3_bucket') # bucket name
s3_prefix = os.environ.get('s3_prefix')
callLogTableName = os.environ.get('callLogTableName')
path = os.environ.get('path')
doc_prefix = s3_prefix+'/'
debugMessageMode = os.environ.get('debugMessageMode', 'false')
agentLangMode = 'kor'
projectName = os.environ.get('projectName')
opensearch_account = os.environ.get('opensearch_account')
opensearch_passwd = os.environ.get('opensearch_passwd')
opensearch_url = os.environ.get('opensearch_url')
LLM_for_chat = json.loads(os.environ.get('LLM_for_chat'))
LLM_for_multimodal= json.loads(os.environ.get('LLM_for_multimodal'))
LLM_embedding = json.loads(os.environ.get('LLM_embedding'))
selected_chat = 0
length_of_models = 1
selected_multimodal = 0
selected_embedding = 0
separated_chat_history = os.environ.get('separated_chat_history')
enableParentDocumentRetrival = os.environ.get('enableParentDocumentRetrival')
enableHybridSearch = os.environ.get('enableHybridSearch')
useParrelWebSearch = True
useEnhancedSearch = True
vectorIndexName = os.environ.get('vectorIndexName')
index_name = vectorIndexName
grade_state = "LLM" # LLM, PRIORITY_SEARCH, OTHERS
numberOfDocs = 2
minDocSimilarity = 400
    
multi_region_models = [   # Nova Pro
    {   
        "bedrock_region": "us-west-2", # Oregon
        "model_type": "nova",
        "model_id": "amazon.nova-pro-v1:0"
    },
    {
        "bedrock_region": "us-east-1", # N.Virginia
        "model_type": "nova",
        "model_id": "amazon.nova-pro-v1:0"
    },
    {
        "bedrock_region": "us-east-2", # Ohio
        "model_type": "nova",
        "model_id": "amazon.nova-pro-v1:0"
    },
    {
        "bedrock_region": "us-west-2", # Oregon
        "model_type": "nova",
        "model_id": "us.amazon.nova-pro-v1:0"
    }
]
multi_region = 'enable'

titan_embedding_v1 = [  # dimension = 1536
  {
    "bedrock_region": "us-west-2", # Oregon
    "model_type": "titan",
    "model_id": "amazon.titan-embed-text-v1"
  },
  {
    "bedrock_region": "us-east-1", # N.Virginia
    "model_type": "titan",
    "model_id": "amazon.titan-embed-text-v1"
  }
]
priority_search_embedding = titan_embedding_v1
selected_ps_embedding = 0

reference_docs = []
# api key to get weather information in agent
secretsmanager = boto3.client('secretsmanager')
try:
    get_weather_api_secret = secretsmanager.get_secret_value(
        SecretId=f"openweathermap-{projectName}"
    )
    #print('get_weather_api_secret: ', get_weather_api_secret)
    secret = json.loads(get_weather_api_secret['SecretString'])
    #print('secret: ', secret)
    weather_api_key = secret['weather_api_key']

except Exception as e:
    raise e
   
# api key to use LangSmith
langsmith_api_key = ""
try:
    get_langsmith_api_secret = secretsmanager.get_secret_value(
        SecretId=f"langsmithapikey-{projectName}"
    )
    #print('get_langsmith_api_secret: ', get_langsmith_api_secret)
    secret = json.loads(get_langsmith_api_secret['SecretString'])
    #print('secret: ', secret)
    langsmith_api_key = secret['langsmith_api_key']
    langchain_project = secret['langchain_project']
except Exception as e:
    raise e

if langsmith_api_key:
    os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = langchain_project
    
# api key to use Tavily Search
tavily_api_key = ""
try:
    get_tavily_api_secret = secretsmanager.get_secret_value(
        SecretId=f"tavilyapikey-{projectName}"
    )
    #print('get_tavily_api_secret: ', get_tavily_api_secret)
    secret = json.loads(get_tavily_api_secret['SecretString'])
    # print('secret: ', secret)
    if secret['tavily_api_key']:
        tavily_api_key = json.loads(secret['tavily_api_key'])
    # print('tavily_api_key: ', tavily_api_key)
except Exception as e: 
    raise e

def check_tavily_secret(tavily_api_key):
    query = 'what is LangGraph'
    valid_keys = []
    for key in tavily_api_key:
        try:
            tavily_client = TavilyClient(api_key=key)
            response = tavily_client.search(query, max_results=1)
            # print('tavily response: ', response)
            
            if 'results' in response and len(response['results']):
                valid_keys.append(key)
        except Exception as e:
            print('Exception: ', e)
    # print('valid_keys: ', valid_keys)
    
    return valid_keys

tavily_api_key = check_tavily_secret(tavily_api_key)
# print('tavily_api_key: ', tavily_api_key)
print('The number of valid tavily api keys: ', len(tavily_api_key))

selected_tavily = -1
if len(tavily_api_key):
    os.environ["TAVILY_API_KEY"] = tavily_api_key[0]
    selected_tavily = 0
      
def tavily_search(query, k):
    global selected_tavily
    docs = []
    
    if selected_tavily != -1:
        selected_tavily = selected_tavily + 1
        if selected_tavily == len(tavily_api_key):
            selected_tavily = 0

        try:
            tavily_client = TavilyClient(api_key=tavily_api_key[selected_tavily])
            response = tavily_client.search(query, max_results=k)
            # print('tavily response: ', response)
            
            for r in response["results"]:
                name = r.get("title")
                if name is None:
                    name = 'WWW'
            
                docs.append(
                    Document(
                        page_content=r.get("content"),
                        metadata={
                            'name': name,
                            'url': r.get("url"),
                            'from': 'tavily'
                        },
                    )
                )   
                
        except Exception as e:
            print('Exception: ', e)
    return docs

# result = tavily_search('what is LangChain', 2)
# print('search result: ', result)

os_client = OpenSearch(
    hosts = [{
        'host': opensearch_url.replace("https://", ""), 
        'port': 443
    }],
    http_compress = True,
    http_auth=(opensearch_account, opensearch_passwd),
    use_ssl = True,
    verify_certs = True,
    ssl_assert_hostname = False,
    ssl_show_warn = False,
)

def reflesh_opensearch_index():
    #########################
    # opensearch index (reflesh)
    #########################
    print(f"deleting opensearch index... {index_name}") 
    
    try: # create index
        response = os_client.indices.delete(
            index_name
        )
        print('opensearch index was deleted:', response)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                
        #raise Exception ("Not able to create the index")        
    return 
    
# websocket
connection_url = os.environ.get('connection_url')
client = boto3.client('apigatewaymanagementapi', endpoint_url=connection_url)
print('connection_url: ', connection_url)

HUMAN_PROMPT = "\n\nHuman:"
AI_PROMPT = "\n\nAssistant:"

map_chain = dict() 
MSG_LENGTH = 100
# Multi-LLM
def get_chat():
    global selected_chat
    
    if multi_region == 'enable':
        profile = multi_region_models[selected_chat]
    else:
        profile = LLM_for_chat[selected_chat]
        
    bedrock_region =  profile['bedrock_region']
    modelId = profile['model_id']
    maxOutputTokens = 4096
    print(f'LLM: {selected_chat}, bedrock_region: {bedrock_region}, modelId: {modelId}')
                          
    # bedrock   
    boto3_bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name=bedrock_region,
        config=Config(
            retries = {
                'max_attempts': 30
            }
        )
    )
    parameters = {
        "max_tokens":maxOutputTokens,     
        "temperature":0.1,
        "top_k":250,
        "top_p":0.9,
        "stop_sequences": [HUMAN_PROMPT]
    }
    # print('parameters: ', parameters)

    chat = ChatBedrock(   # new chat model
        model_id=modelId,
        client=boto3_bedrock, 
        model_kwargs=parameters,
    )    
    
    selected_chat = selected_chat + 1
    if selected_chat == length_of_models:
        selected_chat = 0
    
    return chat

def get_multi_region_chat(models, selected):
    profile = models[selected]
    bedrock_region =  profile['bedrock_region']
    modelId = profile['model_id']
    maxOutputTokens = 4096
    print(f'selected_chat: {selected}, bedrock_region: {bedrock_region}, modelId: {modelId}')
                          
    # bedrock   
    boto3_bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name=bedrock_region,
        config=Config(
            retries = {
                'max_attempts': 30
            }
        )
    )
    parameters = {
        "max_tokens":maxOutputTokens,     
        "temperature":0.1,
        "top_k":250,
        "top_p":0.9,
        "stop_sequences": [HUMAN_PROMPT]
    }
    # print('parameters: ', parameters)

    chat = ChatBedrock(   # new chat model
        model_id=modelId,
        client=boto3_bedrock, 
        model_kwargs=parameters,
    )    
    
    return chat

def get_multimodal():
    global selected_multimodal
    print('LLM_for_chat: ', LLM_for_chat)
    print('selected_multimodal: ', selected_multimodal)
        
    profile = LLM_for_multimodal[selected_multimodal]
    bedrock_region =  profile['bedrock_region']
    modelId = profile['model_id']
    maxOutputTokens = 4096
    print(f'LLM: {selected_multimodal}, bedrock_region: {bedrock_region}, modelId: {modelId}')
                          
    # bedrock   
    boto3_bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name=bedrock_region,
        config=Config(
            retries = {
                'max_attempts': 30
            }
        )
    )
    parameters = {
        "max_tokens":maxOutputTokens,     
        "temperature":0.1,
        "top_k":250,
        "top_p":0.9,
        "stop_sequences": [HUMAN_PROMPT]
    }
    # print('parameters: ', parameters)

    multimodal = ChatBedrock(   # new chat model
        model_id=modelId,
        client=boto3_bedrock, 
        model_kwargs=parameters,
    )    
    
    selected_multimodal = selected_multimodal + 1
    if selected_multimodal == len(LLM_for_multimodal):
        selected_multimodal = 0
    
    return multimodal
    
def get_embedding():
    global selected_embedding
    profile = LLM_embedding[selected_embedding]
    bedrock_region =  profile['bedrock_region']
    model_id = profile['model_id']
    print(f'selected_embedding: {selected_embedding}, bedrock_region: {bedrock_region}, model_id: {model_id}')
    
    # bedrock   
    boto3_bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name=bedrock_region, 
        config=Config(
            retries = {
                'max_attempts': 30
            }
        )
    )
    
    bedrock_embedding = BedrockEmbeddings(
        client=boto3_bedrock,
        region_name = bedrock_region,
        model_id = model_id
    )  
    
    selected_embedding = selected_embedding + 1
    if selected_embedding == len(LLM_embedding):
        selected_embedding = 0
    
    return bedrock_embedding

# load documents from s3 for pdf and txt
def load_document(file_type, s3_file_name):
    s3r = boto3.resource("s3")
    doc = s3r.Object(s3_bucket, s3_prefix+'/'+s3_file_name)
    
    if file_type == 'pdf':
        contents = doc.get()['Body'].read()
        reader = PyPDF2.PdfReader(BytesIO(contents))
        
        raw_text = []
        for page in reader.pages:
            raw_text.append(page.extract_text())
        contents = '\n'.join(raw_text)    
        
    elif file_type == 'txt':        
        contents = doc.get()['Body'].read().decode('utf-8')
        
    print('contents: ', contents)
    new_contents = str(contents).replace("\n"," ") 
    print('length: ', len(new_contents))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""],
        length_function = len,
    ) 

    texts = text_splitter.split_text(new_contents) 
    print('texts[0]: ', texts[0])
    
    return texts

# load csv documents from s3
def load_csv_document(s3_file_name):
    s3r = boto3.resource("s3")
    doc = s3r.Object(s3_bucket, s3_prefix+'/'+s3_file_name)

    lines = doc.get()['Body'].read().decode('utf-8').split('\n')   # read csv per line
    print('lins: ', len(lines))
        
    columns = lines[0].split(',')  # get columns
    #columns = ["Category", "Information"]  
    #columns_to_metadata = ["type","Source"]
    print('columns: ', columns)
    
    docs = []
    n = 0
    for row in csv.DictReader(lines, delimiter=',',quotechar='"'):
        # print('row: ', row)
        #to_metadata = {col: row[col] for col in columns_to_metadata if col in row}
        values = {k: row[k] for k in columns if k in row}
        content = "\n".join(f"{k.strip()}: {v.strip()}" for k, v in values.items())
        doc = Document(
            page_content=content,
            metadata={
                'name': s3_file_name,
                'row': n+1,
            }
            #metadata=to_metadata
        )
        docs.append(doc)
        n = n+1
    print('docs[0]: ', docs[0])

    return docs

def get_summary(chat, docs):    
    text = ""
    for doc in docs:
        text = text + doc
    
    if isKorean(text)==True:
        system = (
            "다음의 <article> tag안의 문장을 요약해서 500자 이내로 설명하세오."
        )
    else: 
        system = (
            "Here is pieces of article, contained in <article> tags. Write a concise summary within 500 characters."
        )
    
    human = "<article>{text}</article>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    # print('prompt: ', prompt)
    
    chain = prompt | chat    
    try: 
        result = chain.invoke(
            {
                "text": text
            }
        )
        
        summary = result.content
        print('result of summarization: ', summary)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                    
        raise Exception ("Not able to request to LLM")
    
    return summary
    
def load_chatHistory(userId, allowTime, chat_memory):
    dynamodb_client = boto3.client('dynamodb')

    response = dynamodb_client.query(
        TableName=callLogTableName,
        KeyConditionExpression='user_id = :userId AND request_time > :allowTime',
        ExpressionAttributeValues={
            ':userId': {'S': userId},
            ':allowTime': {'S': allowTime}
        }
    )
    # print('query result: ', response['Items'])

    for item in response['Items']:
        text = item['body']['S']
        msg = item['msg']['S']
        type = item['type']['S']

        if type == 'text' and text and msg:
            memory_chain.chat_memory.add_user_message(text)
            if len(msg) > MSG_LENGTH:
                memory_chain.chat_memory.add_ai_message(msg[:MSG_LENGTH])                          
            else:
                memory_chain.chat_memory.add_ai_message(msg) 
                
def getAllowTime():
    d = datetime.datetime.now() - datetime.timedelta(days = 2)
    timeStr = str(d)[0:19]
    print('allow time: ',timeStr)

    return timeStr

def isKorean(text):
    # check korean
    pattern_hangul = re.compile('[\u3131-\u3163\uac00-\ud7a3]+')
    word_kor = pattern_hangul.search(str(text))
    # print('word_kor: ', word_kor)

    if word_kor and word_kor != 'None':
        print('Korean: ', word_kor)
        return True
    else:
        print('Not Korean: ', word_kor)
        return False

def general_conversation(connectionId, requestId, chat, query):
    if isKorean(query)==True :
        system = (
            "당신은 사려깊은 인공지능 도우미입니다." 
            "당신은 상황에 맞는 구체적인 세부 정보를 충분히 제공합니다." 
            "너의 이름은 서연이고, 모르는 질문을 받으면 솔직히 모른다고 말합니다."
            "답변은 대화형 text 형태로 제공합니다. markdown 포맷을 사용하지 않습니다."
        )
    else: 
        system = (
            "You will be acting as a thoughtful advisor."
            "Using the following conversation, answer friendly for the newest question." 
            "If you don't know the answer, just say that you don't know, don't try to make up an answer."     
            "You will only answer in text format, using markdown format is not allowed."        
        )
    
    human = "{input}"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), MessagesPlaceholder(variable_name="history"), ("human", human)])
    # print('prompt: ', prompt)
    
    history = memory_chain.load_memory_variables({})["chat_history"]
    # print('memory_chain: ', history)
                
    chain = prompt | chat    
    try: 
        isTyping(connectionId, requestId, "")  
        stream = chain.invoke(
            {
                "history": history,
                "input": query,
            }
        )
        msg = readStreamMsg(connectionId, requestId, stream.content)    
        
        msg = stream.content

    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)        
            
        sendErrorMessage(connectionId, requestId, err_msg)    
        raise Exception ("Not able to request to LLM")
    
    return msg

def get_answer_using_opensearch(chat, text, connectionId, requestId):    
    msg = ""
    top_k = 4
    relevant_docs = []
    
    bedrock_embedding = get_embedding()
       
    vectorstore_opensearch = OpenSearchVectorSearch(
        index_name = index_name,
        is_aoss = False,
        ef_search = 1024, # 512(default)
        m=48,
        #engine="faiss",  # default: nmslib
        embedding_function = bedrock_embedding,
        opensearch_url=opensearch_url,
        http_auth=(opensearch_account, opensearch_passwd), # http_auth=awsauth,
    )  
    
    isTyping(connectionId, requestId, "retrieving...")
    print('multi_region: ', multi_region)
    
    if multi_region == 'enable':  # parallel processing
        relevant_documents = get_documents_from_opensearch(vectorstore_opensearch, text, top_k)
                        
        for i, document in enumerate(relevant_documents):
            print(f'## Document(opensearch-vector) {i+1}: {document}')
            
            parent_doc_id = document[0].metadata['parent_doc_id']
            doc_level = document[0].metadata['doc_level']
            #print(f"child: parent_doc_id: {parent_doc_id}, doc_level: {doc_level}")
            
            content, name, url = get_parent_content(parent_doc_id) # use pareant document
            #print(f"parent_doc_id: {parent_doc_id}, doc_level: {doc_level}, url: {url}, content: {content}")
            
            relevant_docs.append(
                Document(
                    page_content=content,
                    metadata={
                        'name': name,
                        'url': url,
                        'doc_level': doc_level,
                        'from': 'vector'
                    },
                )
            )
    else: 
        print("###### similarity_search_with_score ######")
        relevant_documents = vectorstore_opensearch.similarity_search_with_score(
            query = text,
            k = top_k
        )
        
        for i, document in enumerate(relevant_documents):
            print(f'## Document(opensearch-vector) {i+1}: {document}')
            
            name = document[0].metadata['name']
            url = document[0].metadata['url']
            content = document[0].page_content
                   
            relevant_docs.append(
                Document(
                    page_content=content,
                    metadata={
                        'name': name,
                        'url': url,
                        'from': 'vector'
                    },
                )
            )
    # print('the number of docs (vector search): ', len(relevant_docs))
            
    if enableHybridSearch == 'true':
        relevant_docs += lexical_search(text, top_k)    
        
    isTyping(connectionId, requestId, "grading...")
    
    filtered_docs = grade_documents(text, relevant_docs) # grading
    
    filtered_docs = check_duplication(filtered_docs) # check duplication
            
    relevant_context = ""
    for i, document in enumerate(filtered_docs):
        # print(f"{i}: {document}")
        if document.page_content:
            content = document.page_content
            
        relevant_context = relevant_context + content + "\n\n"
        
    # print('relevant_context: ', relevant_context)

    msg = query_using_RAG_context(connectionId, requestId, chat, relevant_context, text)
               
    return msg

def get_documents_from_opensearch(vectorstore_opensearch, query, top_k):
    print("###### get_documents_from_opensearch ######")
    
    result = vectorstore_opensearch.similarity_search_with_score(
        query = query,
        k = top_k*2,  
        search_type="script_scoring",
        pre_filter={"term": {"metadata.doc_level": "child"}}
    )
    print('result: ', result)
            
    relevant_documents = []
    docList = []
    for re in result:
        if 'parent_doc_id' in re[0].metadata:
            parent_doc_id = re[0].metadata['parent_doc_id']
            doc_level = re[0].metadata['doc_level']
            print(f"doc_level: {doc_level}, parent_doc_id: {parent_doc_id}")
                    
            if doc_level == 'child':
                if parent_doc_id in docList:
                    print('duplicated!')
                else:
                    relevant_documents.append(re)
                    docList.append(parent_doc_id)
                    
                    if len(relevant_documents)>=top_k:
                        break
                                
    # print('relevant_documents: ', relevant_documents)    
    for i, doc in enumerate(relevant_documents):
        if len(doc[0].page_content)>=100:
            text = doc[0].page_content[:100]
        else:
            text = doc[0].page_content            
        print(f"--> vector search doc[{i}]: {text}, metadata:{doc[0].metadata}")
    
    return relevant_documents

def get_parent_content(parent_doc_id):
    response = os_client.get(
        index = index_name, 
        id = parent_doc_id
    )
    
    source = response['_source']                            
    # print('parent_doc: ', source['text'])   
    
    metadata = source['metadata']    
    #print('name: ', metadata['name'])   
    #print('url: ', metadata['url'])   
    #print('doc_level: ', metadata['doc_level']) 
    
    url = ""
    if "url" in metadata:
        url = metadata['url']
    
    return source['text'], metadata['name'], url

@tool 
def get_book_list(keyword: str) -> str:
    """
    Search book list by keyword and then return book list
    keyword: search keyword
    return: book list
    """
    
    keyword = keyword.replace('\'','')

    answer = ""
    url = f"https://search.kyobobook.co.kr/search?keyword={keyword}&gbCode=TOT&target=total"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        prod_info = soup.find_all("a", attrs={"class": "prod_info"})
        
        if len(prod_info):
            answer = "추천 도서는 아래와 같습니다.\n"
            
        for prod in prod_info[:5]:
            title = prod.text.strip().replace("\n", "")       
            link = prod.get("href")
            answer = answer + f"{title}, URL: {link}\n\n"
    
    return answer
    
@tool
def get_current_time(format: str=f"%Y-%m-%d %H:%M:%S")->str:
    """Returns the current date and time in the specified format"""
    # f"%Y-%m-%d %H:%M:%S"
    
    format = format.replace('\'','')
    timestr = datetime.datetime.now(timezone('Asia/Seoul')).strftime(format)
    print('timestr:', timestr)
    
    return timestr

def get_lambda_client(region):
    return boto3.client(
        service_name='lambda',
        region_name=region
    )

@tool    
def get_system_time() -> str:
    """
    retrive system time to earn the current date and time.
    return: a string of date and time
    """    
    
    function_name = "lambda-datetime-for-llm-agent"
    lambda_region = 'ap-northeast-2'
    
    try:
        lambda_client = get_lambda_client(region=lambda_region)
        payload = {}
        print("Payload: ", payload)
            
        response = lambda_client.invoke(
            FunctionName=function_name,
            Payload=json.dumps(payload),
        )
        print("Invoked function %s.", function_name)
        print("Response: ", response)
    except ClientError:
        print("Couldn't invoke function %s.", function_name)
        raise
    
    payload = response['Payload']
    print('payload: ', payload)
    body = json.load(payload)['body']
    print('body: ', body)
    jsonBody = json.loads(body) 
    print('jsonBody: ', jsonBody)    
    timestr = jsonBody['timestr']
    print('timestr: ', timestr)
     
    return timestr

@tool
def get_weather_info(city: str) -> str:
    """
    retrieve weather information by city name and then return weather statement.
    city: the name of city to retrieve
    return: weather statement
    """    
    
    city = city.replace('\n','')
    city = city.replace('\'','')
    city = city.replace('\"','')
                
    chat = get_chat()
    if isKorean(city):
        place = traslation(chat, city, "Korean", "English")
        print('city (translated): ', place)
    else:
        place = city
        city = traslation(chat, city, "English", "Korean")
        print('city (translated): ', city)
        
    print('place: ', place)
    
    weather_str: str = f"{city}에 대한 날씨 정보가 없습니다."
    if weather_api_key: 
        apiKey = weather_api_key
        lang = 'en' 
        units = 'metric' 
        api = f"https://api.openweathermap.org/data/2.5/weather?q={place}&APPID={apiKey}&lang={lang}&units={units}"
        # print('api: ', api)
                
        try:
            result = requests.get(api)
            result = json.loads(result.text)
            print('result: ', result)
        
            if 'weather' in result:
                overall = result['weather'][0]['main']
                current_temp = result['main']['temp']
                min_temp = result['main']['temp_min']
                max_temp = result['main']['temp_max']
                humidity = result['main']['humidity']
                wind_speed = result['wind']['speed']
                cloud = result['clouds']['all']
                
                weather_str = f"{city}의 현재 날씨의 특징은 {overall}이며, 현재 온도는 {current_temp}도 이고, 최저온도는 {min_temp}도, 최고 온도는 {max_temp}도 입니다. 현재 습도는 {humidity}% 이고, 바람은 초당 {wind_speed} 미터 입니다. 구름은 {cloud}% 입니다."
                #weather_str = f"Today, the overall of {city} is {overall}, current temperature is {current_temp} degree, min temperature is {min_temp} degree, highest temperature is {max_temp} degree. huminity is {humidity}%, wind status is {wind_speed} meter per second. the amount of cloud is {cloud}%."            
        except Exception:
            err_msg = traceback.format_exc()
            print('error message: ', err_msg)                    
            # raise Exception ("Not able to request to LLM")    
        
    print('weather_str: ', weather_str)                            
    return weather_str

@tool
def search_by_tavily(keyword: str) -> str:
    """
    Search general information by keyword and then return the result as a string.
    keyword: search keyword
    return: the information of keyword
    """    
    global reference_docs    
    answer = ""
    
    if tavily_api_key:
        keyword = keyword.replace('\'','')
        
        search = TavilySearchResults(
            max_results=3,
            include_answer=True,
            include_raw_content=True,
            search_depth="advanced", # "basic"
            include_domains=["google.com", "naver.com"]
        )
                    
        try: 
            output = search.invoke(keyword)
            print('tavily output: ', output)
            
            for result in output:
                print('result: ', result)
                if result:
                    content = result.get("content")
                    url = result.get("url")
                    
                    reference_docs.append(
                        Document(
                            page_content=content,
                            metadata={
                                'name': 'WWW',
                                'url': url,
                                'from': 'tavily'
                            },
                        )
                    )                
                    answer = answer + f"{content}, URL: {url}\n"
        
        except Exception:
            err_msg = traceback.format_exc()
            print('error message: ', err_msg)                    
            # raise Exception ("Not able to request to tavily")   
        
    return answer

@tool    
def search_by_opensearch(keyword: str) -> str:
    """
    Search technical information by keyword and then return the result as a string.
    keyword: search keyword
    return: the technical information of keyword
    """    
    
    print('keyword: ', keyword)
    keyword = keyword.replace('\'','')
    keyword = keyword.replace('|','')
    keyword = keyword.replace('\n','')
    print('modified keyword: ', keyword)
    
    bedrock_embedding = get_embedding()
        
    vectorstore_opensearch = OpenSearchVectorSearch(
        index_name = index_name,
        is_aoss = False,
        ef_search = 1024, # 512(default)
        m=48,
        #engine="faiss",  # default: nmslib
        embedding_function = bedrock_embedding,
        opensearch_url=opensearch_url,
        http_auth=(opensearch_account, opensearch_passwd), # http_auth=awsauth,
    )     
    top_k = 2
    
    relevant_docs = [] 
    if enableParentDocumentRetrival == 'true': # parent/child chunking
        relevant_documents = get_documents_from_opensearch(vectorstore_opensearch, keyword, top_k)
                        
        for i, document in enumerate(relevant_documents):
            #print(f'## Document(opensearch-vector) {i+1}: {document}')
            
            parent_doc_id = document[0].metadata['parent_doc_id']
            doc_level = document[0].metadata['doc_level']
            #print(f"child: parent_doc_id: {parent_doc_id}, doc_level: {doc_level}")
            
            excerpt, name, url = get_parent_content(parent_doc_id) # use pareant document
            #print(f"parent_doc_id: {parent_doc_id}, doc_level: {doc_level}, url: {url}, content: {excerpt}")
            
            relevant_docs.append(
                Document(
                    page_content=excerpt,
                    metadata={
                        'name': name,
                        'url': url,
                        'doc_level': doc_level,
                        'from': 'vector'
                    },
                )
            )
    else: 
        relevant_documents = vectorstore_opensearch.similarity_search_with_score(
            query = keyword,
            k = top_k,
        )

        for i, document in enumerate(relevant_documents):
            #print(f'## Document(opensearch-vector) {i+1}: {document}')
            
            excerpt = document[0].page_content
            
            url = ""
            if "url" in document[0].metadata:
                url = document[0].metadata['url']
                
            name = document[0].metadata['name']
            
            relevant_docs.append(
                Document(
                    page_content=excerpt,
                    metadata={
                        'name': name,
                        'url': url,
                        'from': 'vector'
                    },
                )
            )
    
    if enableHybridSearch == 'true':
        relevant_docs += lexical_search(keyword, top_k)
    
    print('relevant_docs length: ', len(relevant_docs))
                
    filtered_docs = grade_documents(keyword, relevant_docs)
        
    for i, doc in enumerate(filtered_docs):
        if len(doc.page_content)>=100:
            text = doc.page_content[:100]
        else:
            text = doc.page_content
            
        print(f"filtered doc[{i}]: {text}, metadata:{doc.metadata}")
       
    relevant_docs = "" 
    for doc in filtered_docs:
        content = doc.page_content
        
        relevant_docs = relevant_docs + f"{content}\n\n"
        
    return relevant_docs

def lexical_search(query, top_k):
    # lexical search (keyword)
    min_match = 0
    
    query = {
        "query": {
            "bool": {
                "must": [
                    {
                        "match": {
                            "text": {
                                "query": query,
                                "minimum_should_match": f'{min_match}%',
                                "operator":  "or",
                            }
                        }
                    },
                ],
                "filter": [
                ]
            }
        }
    }

    response = os_client.search(
        body=query,
        index=index_name
    )
    # print('lexical query result: ', json.dumps(response))
        
    docs = []
    for i, document in enumerate(response['hits']['hits']):
        if i>=top_k: 
            break
                    
        excerpt = document['_source']['text']
        
        name = document['_source']['metadata']['name']
        # print('name: ', name)

        page = ""
        if "page" in document['_source']['metadata']:
            page = document['_source']['metadata']['page']
        
        url = ""
        if "url" in document['_source']['metadata']:
            url = document['_source']['metadata']['url']            
        
        docs.append(
                Document(
                    page_content=excerpt,
                    metadata={
                        'name': name,
                        'url': url,
                        'page': page,
                        'from': 'lexical'
                    },
                )
            )
    
    for i, doc in enumerate(docs):
        #print('doc: ', doc)
        #print('doc content: ', doc.page_content)
        
        if len(doc.page_content)>=100:
            text = doc.page_content[:100]
        else:
            text = doc.page_content            
        print(f"--> lexical search doc[{i}]: {text}, metadata:{doc.metadata}")   
        
    return docs

def get_retrieval_grader(chat):
    system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )
    
    structured_llm_grader = chat.with_structured_output(GradeDocuments)
    retrieval_grader = grade_prompt | structured_llm_grader
    return retrieval_grader

def grade_document_based_on_relevance(conn, question, doc, models, selected):     
    chat = get_multi_region_chat(models, selected)
    retrieval_grader = get_retrieval_grader(chat)
    score = retrieval_grader.invoke({"question": question, "document": doc.page_content})
    # print(f"score: {score}")
    
    grade = score.binary_score    
    if grade == 'yes':
        print("---GRADE: DOCUMENT RELEVANT---")
        conn.send(doc)
    else:  # no
        print("---GRADE: DOCUMENT NOT RELEVANT---")
        conn.send(None)
    
    conn.close()
                                    
def grade_documents_using_parallel_processing(question, documents):
    global selected_chat
    
    filtered_docs = []    

    processes = []
    parent_connections = []
    
    for i, doc in enumerate(documents):
        #print(f"grading doc[{i}]: {doc.page_content}")        
        parent_conn, child_conn = Pipe()
        parent_connections.append(parent_conn)
            
        process = Process(target=grade_document_based_on_relevance, args=(child_conn, question, doc, multi_region_models, selected_chat))
        processes.append(process)

        selected_chat = selected_chat + 1
        if selected_chat == length_of_models:
            selected_chat = 0
    for process in processes:
        process.start()
            
    for parent_conn in parent_connections:
        relevant_doc = parent_conn.recv()

        if relevant_doc is not None:
            filtered_docs.append(relevant_doc)

    for process in processes:
        process.join()
    
    #print('filtered_docs: ', filtered_docs)
    return filtered_docs

def web_search_by_tavily(conn, q, k):     
    search = TavilySearchResults(max_results=k) 
    response = search.invoke(q)     
    print('response: ', response)
    
    content = []
    for r in response:
        if 'content' in r:
            content.append(r['content'])
        
    conn.send(content)    
    conn.close()
    
def web_search_using_parallel_processing(qurles):
    content = []    

    processes = []
    parent_connections = []
    
    k = 2
    for i, q in enumerate(qurles):
        parent_conn, child_conn = Pipe()
        parent_connections.append(parent_conn)
                    
        process = Process(target=web_search_by_tavily, args=(child_conn, q, k))
        processes.append(process)
        
    for process in processes:
        process.start()
            
    for parent_conn in parent_connections:
        content += parent_conn.recv()
        
    for process in processes:
        process.join()
    
    return content

def print_doc(i, doc):
    if len(doc.page_content)>=100:
        text = doc.page_content[:100]
    else:
        text = doc.page_content
            
    print(f"{i}: {text}, metadata:{doc.metadata}")
    
def check_duplication(docs):
    length_original = len(docs)
    
    contentList = []
    updated_docs = []
    print('length of relevant_docs:', len(docs))
    for doc in docs:            
        # print('excerpt: ', doc['metadata']['excerpt'])
        if doc.page_content in contentList:
            print('duplicated!')
            continue
        contentList.append(doc.page_content)
        updated_docs.append(doc)            
    length_updateed_docs = len(updated_docs)     
    
    if length_original == length_updateed_docs:
        print('no duplication')
    
    return updated_docs

def get_ps_embedding():
    global selected_ps_embedding
    profile = priority_search_embedding[selected_ps_embedding]
    bedrock_region =  profile['bedrock_region']
    model_id = profile['model_id']
    print(f'selected_ps_embedding: {selected_ps_embedding}, bedrock_region: {bedrock_region}, model_id: {model_id}')
    
    # bedrock   
    boto3_bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name=bedrock_region, 
        config=Config(
            retries = {
                'max_attempts': 30
            }
        )
    )
    
    bedrock_ps_embedding = BedrockEmbeddings(
        client=boto3_bedrock,
        region_name = bedrock_region,
        model_id = model_id
    )  

    selected_ps_embedding = selected_ps_embedding + 1
    if selected_ps_embedding == len(priority_search_embedding):
        selected_ps_embedding = 0

    return bedrock_ps_embedding

def priority_search(query, relevant_docs, minSimilarity):
    excerpts = []
    for i, doc in enumerate(relevant_docs):
        #print('doc: ', doc)

        content = doc.page_content
        # print('content: ', content)

        excerpts.append(
            Document(
                page_content=content,
                metadata={
                    'name': doc.metadata['name'],
                    'url': doc.metadata['url'],
                    'from': doc.metadata['from'],
                    'order':i,
                    'score':0
                }
            )
        )
    #print('excerpts: ', excerpts)

    docs = []
    if len(excerpts):
        embeddings = get_ps_embedding()
        vectorstore_confidence = FAISS.from_documents(
            excerpts,  # documents
            embeddings  # embeddings
        )            
        rel_documents = vectorstore_confidence.similarity_search_with_score(
            query=query,
            k=len(relevant_docs)
        )
        
        for i, document in enumerate(rel_documents):
            print(f'## Document(priority_search) query: {query}, {i+1}: {document}')

            order = document[0].metadata['order']
            name = document[0].metadata['name']
            
            score = document[1]
            print(f"query: {query}, {order}: {name}, {score}")

            relevant_docs[order].metadata['score'] = int(score)

            if score < minSimilarity:
                docs.append(relevant_docs[order])    
        # print('selected docs: ', docs)

    return docs
        
def grade_documents(question, documents):
    print("###### grade_documents ######")
    
    filtered_docs = []
    print("start grading...")
    print("grade_state: ", grade_state)
        
    if grade_state == "LLM":
        if multi_region == 'enable':  # parallel processing
            filtered_docs = grade_documents_using_parallel_processing(question, documents)

        else:
            # Score each doc    
            chat = get_chat()
            retrieval_grader = get_retrieval_grader(chat)
            for i, doc in enumerate(documents):
                # print('doc: ', doc)
                print_doc(i, doc)
                
                score = retrieval_grader.invoke({"question": question, "document": doc.page_content})
                print("score: ", score)
                
                grade = score.binary_score
                print("grade: ", grade)
                # Document relevant
                if grade.lower() == "yes":
                    print("---GRADE: DOCUMENT RELEVANT---")
                    filtered_docs.append(doc)
                # Document not relevant
                else:
                    print("---GRADE: DOCUMENT NOT RELEVANT---")
                    # We do not include the document in filtered_docs
                    # We set a flag to indicate that we want to run web search
                    continue

    elif grade_state == "PRIORITY_SEARCH" and len(documents):
        filtered_docs = priority_search(question, documents, minDocSimilarity)
    else:  # OTHERS
        filtered_docs = documents
    
    global reference_docs 
    reference_docs += filtered_docs    
    # print('langth of reference_docs: ', len(reference_docs))
    
    # print('len(docments): ', len(filtered_docs))    
    return filtered_docs

def print_doc(i, doc):
    if len(doc.page_content)>=100:
        text = doc.page_content[:100]
    else:
        text = doc.page_content
            
    print(f"{i}: {text}, metadata:{doc.metadata}")
    
def get_references(docs):
    reference = "\n\nFrom\n"
    for i, doc in enumerate(docs):
        page = ""
        if "page" in doc.metadata:
            page = doc.metadata['page']
            #print('page: ', page)            
        url = ""
        if "url" in doc.metadata:
            url = doc.metadata['url']
            #print('url: ', url)                
        name = ""
        if "name" in doc.metadata:
            name = doc.metadata['name']
            #print('name: ', name)     
           
        sourceType = ""
        if "from" in doc.metadata:
            sourceType = doc.metadata['from']
        else:
            if useEnhancedSearch:
                sourceType = "OpenSearch"
            else:
                sourceType = "WWW"
        #print('sourceType: ', sourceType)        
        
        #if len(doc.page_content)>=1000:
        #    excerpt = ""+doc.page_content[:1000]
        #else:
        #    excerpt = ""+doc.page_content
        excerpt = ""+doc.page_content
        # print('excerpt: ', excerpt)
        
        # for some of unusual case 
        #excerpt = excerpt.replace('"', '')        
        #excerpt = ''.join(c for c in excerpt if c not in '"')
        excerpt = re.sub('"', '', excerpt)
        # print('excerpt(quotation removed): ', excerpt)
        
        if page:                
            reference = reference + f"{i+1}. {page}page in <a href={url} target=_blank>{name}</a>, {sourceType}, <a href=\"#\" onClick=\"alert(`{excerpt}`)\">관련문서</a>\n"
        else:
            reference = reference + f"{i+1}. <a href={url} target=_blank>{name}</a>, {sourceType}, <a href=\"#\" onClick=\"alert(`{excerpt}`)\">관련문서</a>\n"
    return reference

def retrieve(question):
    # Retrieval
    bedrock_embedding = get_embedding()
        
    vectorstore_opensearch = OpenSearchVectorSearch(
        index_name = index_name,
        is_aoss = False,
        ef_search = 1024, # 512(default)
        m=48,
        #engine="faiss",  # default: nmslib
        embedding_function = bedrock_embedding,
        opensearch_url=opensearch_url,
        http_auth=(opensearch_account, opensearch_passwd), # http_auth=awsauth,
    ) 
    
    top_k = 4
    docs = []    
    if enableParentDocumentRetrival == 'true':
        relevant_documents = get_documents_from_opensearch(vectorstore_opensearch, question, top_k)

        for i, document in enumerate(relevant_documents):
            print(f'## Document(opensearch-vector) {i+1}: {document}')
            
            parent_doc_id = document[0].metadata['parent_doc_id']
            doc_level = document[0].metadata['doc_level']
            print(f"child: parent_doc_id: {parent_doc_id}, doc_level: {doc_level}")
                
            excerpt, name, url = get_parent_content(parent_doc_id) # use pareant document
            print(f"parent_doc_id: {parent_doc_id}, doc_level: {doc_level}, url: {url}, content: {excerpt}")
            
            docs.append(
                Document(
                    page_content=excerpt,
                    metadata={
                        'name': name,
                        'url': url,
                        'doc_level': doc_level,
                        'from': 'vector'
                    },
                )
            )
    else: 
        relevant_documents = vectorstore_opensearch.similarity_search_with_score(
            query = question,
            k = top_k,
        )

        for i, document in enumerate(relevant_documents):
            print(f'## Document(opensearch-vector) {i+1}: {document}')
            
            excerpt = document[0].page_content        
            url = document[0].metadata['url']
                            
            docs.append(
                Document(
                    page_content=excerpt,
                    metadata={
                        'name': name,
                        'url': url,
                        'from': 'vector'
                    },
                )
            )    
    
    if enableHybridSearch=='true':
        docs += lexical_search(question, top_k)
    
    return docs

def web_search(question, documents):
    global reference_docs
    
    # Web search
    web_search_tool = TavilySearchResults(max_results=3)
    
    docs = web_search_tool.invoke({"query": question})
    # print('web_search: ', len(docs))
    
    for d in docs:
        print("d: ", d)
        if 'content' in d:
            web_results = "\n".join(d["content"])
            
    #web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    # print("web_results: ", web_results)
    
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    
    # for reference
    for d in docs:
        content = d.get("content")
        url = d.get("url")
                
        reference_docs.append(
            Document(
                page_content=content,
                metadata={
                    'name': 'WWW',
                    'url': url,
                    'from': 'tavily'
                },
            )
        )
    return documents

def get_reg_chain(langMode):
    if langMode:
        system = (
        """다음의 <context> tag안의 참고자료를 이용하여 상황에 맞는 구체적인 세부 정보를 충분히 제공합니다. Assistant의 이름은 서연이고, 모르는 질문을 받으면 솔직히 모른다고 말합니다.

        <context>
        {context}
        </context>""")
    else: 
        system = (
        """Here is pieces of context, contained in <context> tags. Provide a concise answer to the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        <context>
        {context}
        </context>""")
        
    human = "{question}"
        
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
                    
    chat = get_chat()
    rag_chain = prompt | chat
    return rag_chain

def get_rewrite():
    class RewriteQuestion(BaseModel):
        """rewrited question that is well optimized for retrieval."""

        question: str = Field(description="The new question is optimized to represent semantic intent and meaning of the user")
    
    chat = get_chat()
    structured_llm_rewriter = chat.with_structured_output(RewriteQuestion)
    
    print('langMode: ', langMode)
    
    if langMode:
        system = """당신은 질문 re-writer입니다. 사용자의 의도와 의미을 잘 표현할 수 있도록 질문을 한국어로 re-write하세요."""
    else:
        system = (
            "You a question re-writer that converts an input question to a better version that is optimized"
            "for web search. Look at the input and try to reason about the underlying semantic intent / meaning."
        )
        
    print('system: ', system)
        
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Question: {question}"),
        ]
    )
    question_rewriter = re_write_prompt | structured_llm_rewriter
    return question_rewriter

def get_answer_grader():
    class GradeAnswer(BaseModel):
        """Binary score to assess answer addresses question."""

        binary_score: str = Field(
            description="Answer addresses the question, 'yes' or 'no'"
        )
        
    chat = get_chat()
    structured_llm_grade_answer = chat.with_structured_output(GradeAnswer)
        
    system = (
        "You are a grader assessing whether an answer addresses / resolves a question."
        "Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."
    )
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
        ]
    )
    answer_grader = answer_prompt | structured_llm_grade_answer
    return answer_grader

@tool
def grade_answer_for_tool(question: str, answer: str):
    """
    Grade whether the answer is useful or not
    keyword: question and generated answer which could be useful
    return: binary score represented by "yes" or "no"
    """    
    print("###### grade_answer ######")
    print('question: ', question)
    print('answer: ', answer)
                
    answer_grader = get_answer_grader()    
    score = answer_grader.invoke({"question": question, "generation": answer})
    answer_grade = score.binary_score        
    print("answer_grade: ", answer_grade)

    if answer_grade == "yes":
        print("---DECISION: GENERATION ADDRESSES QUESTION---")
        result_str = f"This answer was varified: {answer}"
        return (
            result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
        )
    else:
        print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
        return f"This answer is invalid. Try again from retrieval_node"
    
def get_hallucination_grader():    
    class GradeHallucinations(BaseModel):
        """Binary score for hallucination present in generation answer."""

        binary_score: str = Field(
            description="Answer is grounded in the facts, 'yes' or 'no'"
        )
        
    system = (
        "You are a grader assessing whether an LLM generation is grounded in supported by a set of retrieved facts."
        "Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in supported by the set of facts."
    )    
    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
        ]
    )
        
    chat = get_chat()
    structured_llm_grade_hallucination = chat.with_structured_output(GradeHallucinations)
        
    hallucination_grader = hallucination_prompt | structured_llm_grade_hallucination
    return hallucination_grader
    
# define tools
tools = [get_current_time, get_book_list, get_weather_info, search_by_tavily, search_by_opensearch]        

def update_state_message(msg:str, config):
    print(msg)
    # print('config: ', config)
    
    requestId = config.get("configurable", {}).get("requestId", "")
    connectionId = config.get("configurable", {}).get("connectionId", "")
    
    isTyping(connectionId, requestId, msg)
    
def init_enhanced_search():
    chat = get_chat() 

    model = chat.bind_tools(tools)

    class State(TypedDict):
        messages: Annotated[list, add_messages]

    tool_node = ToolNode(tools)

    def should_continue(state: State) -> Literal["continue", "end"]:
        messages = state["messages"]    
        # print('(should_continue) messages: ', messages)
            
        last_message = messages[-1]
        if not last_message.tool_calls:
            return "end"
        else:                
            return "continue"

    def call_model(state: State, config):
        question = state["messages"]
        print('question: ', question)
        
        update_state_message("thinking...", config)
            
        if isKorean(question[0].content)==True:
            system = (
                "Assistant는 질문에 답변하기 위한 정보를 수집하는 연구원입니다."
                "Assistant은 상황에 맞는 구체적인 세부 정보를 충분히 제공합니다."
                "Assistant는 모르는 질문을 받으면 솔직히 모른다고 말합니다."
                "최종 답변에는 조사한 내용을 반드시 포함하여야 하고, <result> tag를 붙여주세요."
            )
        else: 
            system = (            
                "You are a researcher charged with providing information that can be used when making answer."
                "If you don't know the answer, just say that you don't know, don't try to make up an answer."
                "You will be acting as a thoughtful advisor."
                "Put it in <result> tags."
            )
                
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        chain = prompt | model
                
        response = chain.invoke(question)
        print('call_model response: ', response.tool_calls)
        
        # state messag
        if response.tool_calls:
            toolinfo = response.tool_calls[-1]            
            if toolinfo['type'] == 'tool_call':
                print('tool name: ', toolinfo['name'])                    
                update_state_message(f"calling... {toolinfo['name']}", config)
        
        return {"messages": [response]}

    def buildChatAgent():
        workflow = StateGraph(State)

        workflow.add_node("agent", call_model)
        workflow.add_node("action", tool_node)
            
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "continue": "action",
                "end": END,
            },
        )
        workflow.add_edge("action", "agent")
        return workflow.compile()
    
    return buildChatAgent()

app_enhanced_search = init_enhanced_search()

def enhanced_search(query, config):
    print("###### enhanced_search ######")
    inputs = [HumanMessage(content=query)]
        
    result = app_enhanced_search.invoke({"messages": inputs}, config)   
    print('result: ', result)
            
    message = result["messages"][-1]
    print('enhanced_search: ', message)

    return message.content[message.content.find('<result>')+8:len(message.content)-9]

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

####################### LangGraph #######################
# Chat Agent Executor 
# Agentic Workflow: Tool Use
#########################################################
def run_agent_executor2(connectionId, requestId, query):        
    class State(TypedDict):
        messages: Annotated[list, add_messages]
        sender: str

    tool_node = ToolNode(tools)
            
    def create_agent(chat, tools, system_message: str):        
        tool_names = ", ".join([tool.name for tool in tools])
        print("tool_names: ", tool_names)
        
        system = (
            "당신의 이름은 서연이고, 질문에 친근한 방식으로 대답하도록 설계된 대화형 AI입니다."
            "상황에 맞는 구체적인 세부 정보를 충분히 제공합니다."
            "모르는 질문을 받으면 솔직히 모른다고 말합니다."
            "최종 답변에는 조사한 내용을 반드시 포함합니다."
            #"최종 답변에는 조사한 내용을 반드시 포함하여야 하고, <result> tag를 붙여주세요."         
            "You are a helpful AI assistant, collaborating with other assistants."
            "Use the provided tools to progress towards answering the question."
            "If you are unable to fully answer, that's OK, another assistant with different tools "
            "will help where you left off. Execute what you can to make progress."
            #"If you or any of the other assistants have the final answer or deliverable,"
            #"prefix your response with FINAL ANSWER so the team knows to stop."
            "You have access to the following tools: {tool_names}."
            "{system_message}"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system",system),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        
        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=tool_names)
        
        return prompt | chat.bind_tools(tools)
    
    def agent_node(state, agent, name):
        print(f"###### agent_node:{name} ######")        
        print('state: ', state["messages"])
        
        response = agent.invoke(state["messages"])
        print('response: ', response)
                
        # We convert the agent output into a format that is suitable to append to the global state
        if isinstance(response, ToolMessage):
            pass
        else:
            response = AIMessage(**response.dict(exclude={"type", "name"}), name=name)            
            
        return {
            "messages": [response],
            "sender": name,
        }
    
    chat = get_chat()
    #system_message = "You should provide accurate data for the chart_generator to use."
    system_message = "You should provide accurate data for the questione."
    execution_agent = create_agent(chat, tools, system_message)
    
    execution_agent_node = functools.partial(agent_node, agent=execution_agent, name="execution_agent")
    
    def should_continue(state: State) -> Literal["continue", "end"]:
        print("###### should_continue ######")
        messages = state["messages"]    
        # print('(should_continue) messages: ', messages)
        
        last_message = messages[-1]        
        if not last_message.tool_calls:
            print("Final: ", last_message.content)
            print("--- END ---")
            return "end"
        else:      
            print(f"tool_calls: ", last_message.tool_calls)            
            print(f"--- CONTINUE: {last_message.tool_calls[-1]['name']} ---")
            return "continue"

    def buildAgentExecutor():
        workflow = StateGraph(State)

        workflow.add_node("agent", execution_agent_node)
        workflow.add_node("action", tool_node)
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "continue": "action",
                "end": END,
            },
        )
        workflow.add_edge("action", "agent")

        return workflow.compile()

    app = buildAgentExecutor()
        
    isTyping(connectionId, requestId, "")
    
    inputs = [HumanMessage(content=query)]
    config = {
        "recursion_limit": 50,
        "requestId": requestId,
        "connectionId": connectionId
    }
    
    message = ""
    for event in app.stream({"messages": inputs}, config, stream_mode="values"):   
        # print('event: ', event)
        
        message = event["messages"][-1]
        # print('message: ', message)

    msg = readStreamMsg(connectionId, requestId, message.content)

    #return msg[msg.find('<result>')+8:len(msg)-9]
    return msg

####################### LangGraph #######################
# Corrective RAG
#########################################################
langMode = False

def run_corrective_rag(connectionId, requestId, query):
    class State(TypedDict):
        question : str
        generation : str
        web_search : str
        documents : List[str]

    def retrieve_node(state: State):
        print("###### retrieve ######")
        question = state["question"]
        
        docs = retrieve(question)
        
        return {"documents": docs, "question": question}

    def grade_documents_node(state: State, config):
        print("###### grade_documents ######")
        question = state["question"]
        documents = state["documents"]
        
        update_state_message("grading...", config)
        
        # Score each doc
        filtered_docs = []
        print("start grading...")
        print("grade_state: ", grade_state)
        
        web_search = "No"
        
        if grade_state == "LLM":
            if multi_region == 'enable':  # parallel processing            
                filtered_docs = grade_documents_using_parallel_processing(question, documents)
                
                if len(documents) != len(filtered_docs):
                    web_search = "Yes"

            else:    
                chat = get_chat()
                retrieval_grader = get_retrieval_grader(chat)
                for doc in documents:
                    score = retrieval_grader.invoke({"question": question, "document": doc.page_content})
                    grade = score.binary_score
                    # Document relevant
                    if grade.lower() == "yes":
                        print("---GRADE: DOCUMENT RELEVANT---")
                        filtered_docs.append(doc)
                    # Document not relevant
                    else:
                        print("---GRADE: DOCUMENT NOT RELEVANT---")
                        # We do not include the document in filtered_docs
                        # We set a flag to indicate that we want to run web search
                        web_search = "Yes"
                        continue
            print('len(docments): ', len(filtered_docs))
            print('web_search: ', web_search)
            
        elif grade_state == "PRIORITY_SEARCH":
            filtered_docs = priority_search(question, documents, minDocSimilarity)
        else:  # OTHERS
            filtered_docs = documents
        
        global reference_docs
        reference_docs += filtered_docs
        
        return {"question": question, "documents": filtered_docs, "web_search": web_search}

    def decide_to_generate(state: State):
        print("###### decide_to_generate ######")
        web_search = state["web_search"]
        
        if web_search == "Yes":
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---")
            return "rewrite"
        else:
            # We have relevant documents, so generate answer
            print("---DECISION: GENERATE---")
            return "generate"

    def generate_node(state: State, config):
        print("###### generate ######")
        question = state["question"]
        documents = state["documents"]
        
        update_state_message("generating...", config)
        
        # RAG generation
        rag_chain = get_reg_chain(isKorean(question))
        
        generation = rag_chain.invoke({"context": documents, "question": question})
        print('generation: ', generation.content)
            
        return {"documents": documents, "question": question, "generation": generation}

    def rewrite_node(state: State, config):
        print("###### rewrite ######")
        question = state["question"]
        documents = state["documents"]
        
        update_state_message("rewriting...", config)

        # Prompt
        question_rewriter = get_rewrite()
        
        better_question = question_rewriter.invoke({"question": question})
        print("better_question: ", better_question.question)

        return {"question": better_question.question, "documents": documents}

    def web_search_node(state: State, config):
        print("###### web_search ######")
        question = state["question"]
        documents = state["documents"]
        
        update_state_message("web searching...", config)

        documents = web_search(question, documents)
            
        return {"question": question, "documents": documents}

    def buildCorrectiveRAG():
        workflow = StateGraph(State)
            
        # Define the nodes
        workflow.add_node("retrieve", retrieve_node)  
        workflow.add_node("grade_documents", grade_documents_node)
        workflow.add_node("generate", generate_node)
        workflow.add_node("rewrite", rewrite_node)
        workflow.add_node("websearch", web_search_node)

        # Build graph
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            decide_to_generate,
            {
                "rewrite": "rewrite",
                "generate": "generate",
            },
        )
        workflow.add_edge("rewrite", "websearch")
        workflow.add_edge("websearch", "generate")
        workflow.add_edge("generate", END)

        return workflow.compile()

    app = buildCorrectiveRAG()
    
    global langMode
    langMode = isKorean(query)
            
    isTyping(connectionId, requestId, "")
    
    inputs = {"question": query}
    config = {
        "recursion_limit": 50,
        "requestId": requestId,
        "connectionId": connectionId
    }
    
    for output in app.stream(inputs, config):   
        for key, value in output.items():
            print(f"Finished running: {key}")
            # print("value: ", value)
            
    #print('value: ', value)
        
    readStreamMsg(connectionId, requestId, value["generation"].content)
    
    return value["generation"].content

####################### LangGraph #######################
# Self RAG
#########################################################
MAX_RETRIES = 2 # total 3

class GraphConfig(TypedDict):
    max_retries: int    
    max_count: int

def run_self_rag(connectionId, requestId, query):
    class State(TypedDict):
        question : str
        generation : str
        retries: int  # number of generation 
        count: int # number of retrieval
        documents : List[str]
    
    def retrieve_node(state: State, config):
        print('state: ', state)
        print("###### retrieve ######")
        question = state["question"]
        
        update_state_message("retrieving...", config)
        
        docs = retrieve(question)
        
        return {"documents": docs, "question": question}
    
    def generate_node(state: State, config):
        print("###### generate ######")
        question = state["question"]
        documents = state["documents"]
        retries = state["retries"] if state.get("retries") is not None else -1
        
        update_state_message("generating...", config)
        
        # RAG generation
        rag_chain = get_reg_chain(isKorean(question))
        
        generation = rag_chain.invoke({"context": documents, "question": question})
        print('generation: ', generation.content)
        
        return {"documents": documents, "question": question, "generation": generation, "retries": retries + 1}
            
    def grade_documents_node(state: State, config):
        print("###### grade_documents ######")
        question = state["question"]
        documents = state["documents"]
        count = state["count"] if state.get("count") is not None else -1
        
        print("start grading...")
        print("grade_state: ", grade_state)
        update_state_message("grading...", config)
        
        if grade_state == "LLM":
            if multi_region == 'enable':  # parallel processing            
                filtered_docs = grade_documents_using_parallel_processing(question, documents)

            else:    
                # Score each doc
                filtered_docs = []
                chat = get_chat()
                retrieval_grader = get_retrieval_grader(chat)
                for doc in documents:
                    score = retrieval_grader.invoke({"question": question, "document": doc.page_content})
                    grade = score.binary_score
                    # Document relevant
                    if grade.lower() == "yes":
                        print("---GRADE: DOCUMENT RELEVANT---")
                        filtered_docs.append(doc)
                    # Document not relevant
                    else:
                        print("---GRADE: DOCUMENT NOT RELEVANT---")
                        # We do not include the document in filtered_docs
                        # We set a flag to indicate that we want to run web search
                        continue
            print('len(docments): ', len(filtered_docs))    

        elif grade_state == "PRIORITY_SEARCH":
            filtered_docs = priority_search(question, documents, minDocSimilarity)
        else:  # OTHERS
            filtered_docs = documents
        
        global reference_docs
        reference_docs += filtered_docs
        
        return {"question": question, "documents": filtered_docs, "count": count + 1}

    def decide_to_generate(state: State, config):
        print("###### decide_to_generate ######")
        filtered_documents = state["documents"]
        
        count = state["count"] if state.get("count") is not None else -1
        max_count = config.get("configurable", {}).get("max_count", MAX_RETRIES)
        print("count: ", count)
        
        if not filtered_documents:
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---")
            return "no document" if count < max_count else "not available"
        else:
            # We have relevant documents, so generate answer
            print("---DECISION: GENERATE---")
            return "document"

    def rewrite_node(state: State, config):
        print("###### rewrite ######")
        question = state["question"]
        documents = state["documents"]
        
        update_state_message("rewriting...", config)

        # Prompt
        question_rewriter = get_rewrite()
        
        better_question = question_rewriter.invoke({"question": question})
        print("better_question: ", better_question.question)

        return {"question": better_question.question, "documents": documents}

    def grade_generation(state: State, config):
        print("###### grade_generation ######")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
        
        update_state_message("grading...", config)
        
        retries = state["retries"] if state.get("retries") is not None else -1
        max_retries = config.get("configurable", {}).get("max_retries", MAX_RETRIES)

        hallucination_grader = get_hallucination_grader()
        score = hallucination_grader.invoke(
            {"documents": documents, "generation": generation}
        )
        hallucination_grade = score.binary_score
        
        print("hallucination_grade: ", hallucination_grade)
        print("retries: ", retries)

        # Check hallucination
        answer_grader = get_answer_grader()    
        if hallucination_grade == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            # Check question-answering
            print("---GRADE GENERATION vs QUESTION---")
            score = answer_grader.invoke({"question": question, "generation": generation})
            answer_grade = score.binary_score        
            # print("answer_grade: ", answer_grade)

            if answer_grade == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful" 
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful" if retries < max_retries else "not available"
        else:
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported" if retries < max_retries else "not available"
        
    def build():
        workflow = StateGraph(State)
            
        # Define the nodes
        workflow.add_node("retrieve", retrieve_node)  
        workflow.add_node("grade_documents", grade_documents_node)
        workflow.add_node("generate", generate_node)
        workflow.add_node("rewrite", rewrite_node)

        # Build graph
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            decide_to_generate,
            {
                "no document": "rewrite",
                "document": "generate",
                "not available": "generate",
            },
        )
        workflow.add_edge("rewrite", "retrieve")
        workflow.add_conditional_edges(
            "generate",
            grade_generation,
            {
                "not supported": "generate",
                "useful": END,
                "not useful": "rewrite",
                "not available": END,
            },
        )

        return workflow.compile()

    app = build()    
    
    global langMode
    langMode = isKorean(query)
    
    isTyping(connectionId, requestId, "")
    
    inputs = {"question": query}
    config = {
        "recursion_limit": 50,
        "requestId": requestId,
        "connectionId": connectionId
    }
    
    for output in app.stream(inputs, config):   
        for key, value in output.items():
            print(f"Finished running: {key}")
            # print("value: ", value)
            
    #print('value: ', value)
        
    readStreamMsg(connectionId, requestId, value["generation"].content)
    
    return value["generation"].content

####################### LangGraph #######################
# Self-Corrective RAG
#########################################################
def run_self_corrective_rag(connectionId, requestId, query):
    class State(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]
        question: str
        documents: list[Document]
        candidate_answer: str
        retries: int
        web_fallback: bool

    def retrieve_node(state: State, config):
        print("###### retrieve ######")
        question = state["question"]
        
        update_state_message("retrieving...", config)
        
        docs = retrieve(question)
        
        return {"documents": docs, "question": question, "web_fallback": True}

    def generate_node(state: State, config):
        print("###### generate ######")
        question = state["question"]
        documents = state["documents"]
        retries = state["retries"] if state.get("retries") is not None else -1
        
        update_state_message("generating...", config)
        
        # RAG generation
        rag_chain = get_reg_chain(isKorean(question))
        
        generation = rag_chain.invoke({"context": documents, "question": question})
        print('generation: ', generation.content)
        
        global reference_docs
        reference_docs += documents
        
        return {"retries": retries + 1, "candidate_answer": generation.content}

    def rewrite_node(state: State, config):
        print("###### rewrite ######")
        question = state["question"]
        documents = state["documents"]

        update_state_message("rewriting...", config)
        
        # Prompt
        question_rewriter = get_rewrite()
        
        better_question = question_rewriter.invoke({"question": question})
        print("better_question: ", better_question.question)

        return {"question": better_question.question, "documents": documents}
    
    def grade_generation(state: State, config):
        print("###### grade_generation ######")
        question = state["question"]
        documents = state["documents"]
        generation = state["candidate_answer"]
        web_fallback = state["web_fallback"]
        
        update_state_message("grading...", config)
        
        retries = state["retries"] if state.get("retries") is not None else -1
        max_retries = config.get("configurable", {}).get("max_retries", MAX_RETRIES)

        if not web_fallback:
            return "finalize_response"
        
        print("---Hallucination?---")    
        hallucination_grader = get_hallucination_grader()
        score = hallucination_grader.invoke(
            {"documents": documents, "generation": generation}
        )
        hallucination_grade = score.binary_score
            
        # Check hallucination
        if hallucination_grade == "no":
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS (Hallucination), RE-TRY---")
            return "generate" if retries < max_retries else "websearch"

        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")

        # Check question-answering
        answer_grader = get_answer_grader()    
        score = answer_grader.invoke({"question": question, "generation": generation})
        answer_grade = score.binary_score     
        print("answer_grade: ", answer_grade)
        
        if answer_grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "finalize_response"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION (Not Answer)---")
            return "rewrite" if retries < max_retries else "websearch"

    def web_search_node(state: State, config):
        print("###### web_search ######")
        question = state["question"]
        documents = state["documents"]
        
        update_state_message("web searching...", config)

        documents = web_search(question, documents)
            
        return {"question": question, "documents": documents}

    def finalize_response_node(state: State):
        print("###### finalize_response ######")
        return {"messages": [AIMessage(content=state["candidate_answer"])]}
        
    def buildSelCorrectivefRAG():
        workflow = StateGraph(State)
            
        # Define the nodes
        workflow.add_node("retrieve", retrieve_node)  
        workflow.add_node("generate", generate_node) 
        workflow.add_node("rewrite", rewrite_node)
        workflow.add_node("websearch", web_search_node)
        workflow.add_node("finalize_response", finalize_response_node)

        # Build graph
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("rewrite", "retrieve")
        workflow.add_edge("websearch", "generate")
        workflow.add_edge("finalize_response", END)

        workflow.add_conditional_edges(
            "generate",
            grade_generation,
            {
                "generate": "generate",
                "websearch": "websearch",
                "rewrite": "rewrite",
                "finalize_response": "finalize_response",
            },
        )

        # Compile
        return workflow.compile()

    app = buildSelCorrectivefRAG()

    global langMode
    langMode = isKorean(query)
    
    isTyping(connectionId, requestId, "")
    
    inputs = {"question": query}
    config = {
        "recursion_limit": 50,
        "requestId": requestId,
        "connectionId": connectionId
    }
    
    for output in app.stream(inputs, config):   
        for key, value in output.items():
            print(f"Finished running: {key}")
            #print("value: ", value)
            
    #print('value: ', value)
    #print('content: ', value["messages"][-1].content)
        
    readStreamMsg(connectionId, requestId, value["messages"][-1].content)
    
    return value["messages"][-1].content

####################### LangGraph #######################
# Plan and Execute
#########################################################
def run_plan_and_exeucute(connectionId, requestId, query):
    class State(TypedDict):
        input: str
        plan: list[str]
        past_steps: Annotated[List[Tuple], operator.add]
        info: Annotated[List[Tuple], operator.add]
        answer: str

    class Plan(BaseModel):
        """List of steps as a json format"""

        steps: List[str] = Field(
            description="different steps to follow, should be in sorted order"
        )

    def get_planner():
        system = (
            "For the given objective, come up with a simple step by step plan."
            "This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps."
            "The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps."
        )
            
        planner_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("placeholder", "{messages}"),
            ]
        )
        
        chat = get_chat()   
        
        planner = planner_prompt | chat
        return planner
    
    def plan_node(state: State, config):
        print("###### plan ######")
        print('input: ', state["input"])
        
        update_state_message("planning...", config)
        
        inputs = [HumanMessage(content=state["input"])]

        planner = get_planner()
        response = planner.invoke({"messages": inputs})
        print('response.content: ', response.content)
        
        for attempt in range(5):
            chat = get_chat()
            structured_llm = chat.with_structured_output(Plan, include_raw=True)
            info = structured_llm.invoke(response.content)
            print(f'attempt: {attempt}, info: {info}')
            
            if not info['parsed'] == None:
                parsed_info = info['parsed']
                # print('parsed_info: ', parsed_info)        
                print('steps: ', parsed_info.steps)                
                return {
                    "input": state["input"],
                    "plan": parsed_info.steps
                }
        
        print('parsing_error: ', info['parsing_error'])
        return {"plan": []}          

    def execute_node(state: State, config):
        print("###### execute ######")
        print('input: ', state["input"])
        plan = state["plan"]
        print('plan: ', plan) 
        
        update_state_message("executing...", config)
        
        plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
        #print("plan_str: ", plan_str)
        
        task = plan[0]
        task_formatted = f"""For the following plan:{plan_str}\n\nYou are tasked with executing step {1}, {task}."""
        # print("request: ", task_formatted)     
        request = HumanMessage(content=task_formatted)
        
        chat = get_chat()
        prompt = ChatPromptTemplate.from_messages([
            (
                "system", (
                    "당신의 이름은 서연이고, 질문에 친근한 방식으로 대답하도록 설계된 대화형 AI입니다."
                    "상황에 맞는 구체적인 세부 정보를 충분히 제공합니다."
                    "모르는 질문을 받으면 솔직히 모른다고 말합니다."
                    "결과는 <result> tag를 붙여주세요."
                )
            ),
            MessagesPlaceholder(variable_name="messages"),
        ])
        chain = prompt | chat
        
        response = chain.invoke({"messages": [request]})
        result = response.content
        output = result[result.find('<result>')+8:len(result)-9] # remove <result> tag
        
        print('task: ', task)
        print('executor output: ', output)
        
        # print('plan: ', state["plan"])
        # print('past_steps: ', task)        
        return {
            "input": state["input"],
            "plan": state["plan"],
            "info": [output],
            "past_steps": [task],
        }

    class Response(BaseModel):
        """Response to user."""
        response: str
        
    class Act(BaseModel):
        """Action to perform as a json format"""
        action: Union[Response, Plan] = Field(
            description="Action to perform. If you want to respond to user, use Response. "
            "If you need to further use tools to get the answer, use Plan."
        )
        
    def get_replanner():
        replanner_prompt = ChatPromptTemplate.from_template(
            "For the given objective, come up with a simple step by step plan."
            "This plan should involve individual tasks, that if executed correctly will yield the correct answer."
            "Do not add any superfluous steps."
            "The result of the final step should be the final answer."
            "Make sure that each step has all the information needed - do not skip steps."

            "Your objective was this:"
            "{input}"

            "Your original plan was this:"
            "{plan}"

            "You have currently done the follow steps:"
            "{past_steps}"

            "Update your plan accordingly."
            "If no more steps are needed and you can return to the user, then respond with that."
            "Otherwise, fill out the plan."
            "Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan."
        )
        
        chat = get_chat()
        replanner = replanner_prompt | chat
        
        return replanner

    def replan_node(state: State, config):
        print('#### replan ####')
        
        update_state_message("replanning...", config)
        
        replanner = get_replanner()
        output = replanner.invoke(state)
        print('replanner output: ', output.content)
        
        result = None
        for attempt in range(5):
            chat = get_chat()
            structured_llm = chat.with_structured_output(Act, include_raw=True)    
            info = structured_llm.invoke(output.content)
            print(f'attempt: {attempt}, info: {info}')
            
            if not info['parsed'] == None:
                result = info['parsed']
                print('act output: ', result)
                break
                    
        if result == None:
            return {"response": "답을 찾지 못하였습니다. 다시 해주세요."}
        else:
            if isinstance(result.action, Response):
                return {
                    "response": result.action.response,
                    "info": [result.action.response]
                }
            else:
                return {"plan": result.action.steps}
        
    def should_end(state: State) -> Literal["continue", "end"]:
        print('#### should_end ####')
        print('state: ', state)
        if "response" in state and state["response"]:
            return "end"
        else:
            return "continue"    
        
    def final_answer(state: State) -> str:
        print('#### final_answer ####')
        
        # get final answer
        context = state['info']
        print('context: ', context)
        
        query = state['input']
        print('query: ', query)
        
        if isKorean(query)==True:
            system = (
                "Assistant의 이름은 서연이고, 질문에 대해 친절하게 답변하는 도우미입니다."
                "다음의 <context> tag안의 참고자료를 이용하여 질문에 대한 답변합니다."
                "답변의 이유를 풀어서 명확하게 설명합니다."
                "결과는 <result> tag를 붙여주세요."
                
                "<context>"
                "{context}"
                "</context>"
            )
        else: 
            system = (
                "Here is pieces of context, contained in <context> tags."
                "Provide a concise answer to the question at the end."
                "Explains clearly the reason for the answer."
                "If you don't know the answer, just say that you don't know, don't try to make up an answer."
                "Put it in <result> tags."
                
                "<context>"
                "{context}"
                "</context>"
            )
    
        human = "{input}"
        
        prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
        # print('prompt: ', prompt)
                    
        chat = get_chat()
        chain = prompt | chat
        
        try: 
            response = chain.invoke(
                {
                    "context": context,
                    "input": query,
                }
            )
            result = response.content
            output = result[result.find('<result>')+8:len(result)-9] # remove <result> tag
            print('output: ', output)
            
        except Exception:
            err_msg = traceback.format_exc()
            print('error message: ', err_msg)      
            
        return {"answer": output}  

    def buildPlanAndExecute():
        workflow = StateGraph(State)
        workflow.add_node("planner", plan_node)
        workflow.add_node("executor", execute_node)
        workflow.add_node("replaner", replan_node)
        workflow.add_node("final_answer", final_answer)
        
        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "executor")
        workflow.add_edge("executor", "replaner")
        workflow.add_conditional_edges(
            "replaner",
            should_end,
            {
                "continue": "executor",
                "end": "final_answer",
            },
        )
        workflow.add_edge("final_answer", END)

        return workflow.compile()

    app = buildPlanAndExecute()    
    
    isTyping(connectionId, requestId, "")
    
    inputs = {"input": query}
    config = {
        "recursion_limit": 50,
        "requestId": requestId,
        "connectionId": connectionId
    }
    
    for output in app.stream(inputs, config):   
        for key, value in output.items():
            print(f"Finished: {key}")
            #print("value: ", value)            
    print('value: ', value)
    
    readStreamMsg(connectionId, requestId, value["answer"])
        
    return value["answer"]

####################### LangGraph #######################
# Knowledge Guru
#########################################################
def run_knowledge_guru(connectionId, requestId, query):
    class State(TypedDict):
        messages: Annotated[list, add_messages]
        reflection: list
        search_queries: list
            
    def generate(state: State, config):    
        print("###### generate ######")
        print('state: ', state["messages"])
        print('task: ', state['messages'][0].content)
        
        update_state_message("generating...", config)
        
        draft = enhanced_search(state['messages'][0].content, config)  
        print('draft: ', draft)
        
        return {
            "messages": [AIMessage(content=draft)]
        }
    
    class Reflection(BaseModel):
        missing: str = Field(description="Critique of what is missing.")
        advisable: str = Field(description="Critique of what is helpful for better answer")
        superfluous: str = Field(description="Critique of what is superfluous")

    class Research(BaseModel):
        """Provide reflection and then follow up with search queries to improve the answer."""

        reflection: Reflection = Field(description="Your reflection on the initial answer.")
        search_queries: list[str] = Field(
            description="1-3 search queries for researching improvements to address the critique of your current answer."
        )
    
    def reflect(state: State, config):
        print("###### reflect ######")
        print('state: ', state["messages"])    
        print('draft: ', state["messages"][-1].content)
        
        update_state_message("reflecting...", config)
    
        reflection = []
        search_queries = []
        for attempt in range(5):
            chat = get_chat()
            structured_llm = chat.with_structured_output(Research, include_raw=True)
            
            info = structured_llm.invoke(state["messages"][-1].content)
            print(f'attempt: {attempt}, info: {info}')
                
            if not info['parsed'] == None:
                parsed_info = info['parsed']
                # print('reflection: ', parsed_info.reflection)                
                reflection = [parsed_info.reflection.missing, parsed_info.reflection.advisable]
                search_queries = parsed_info.search_queries
                
                print('reflection: ', parsed_info.reflection)            
                print('search_queries: ', search_queries)                
                break
        
        return {
            "messages": state["messages"],
            "reflection": reflection,
            "search_queries": search_queries
        }

    def revise_answer(state: State, config):   
        print("###### revise_answer ######")
        
        update_state_message("revising...", config)
        
        system = """Revise your previous answer using the new information. 
You should use the previous critique to add important information to your answer. provide the final answer with <result> tag. 
<critique>
{reflection}
</critique>

<information>
{content}
</information>"""
                    
        reflection_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
            
        content = []        
        if useEnhancedSearch:
            for q in state["search_queries"]:
                response = enhanced_search(q, config)
                print(f'q: {q}, response: {response}')
                content.append(response)                   
        else:
            search = TavilySearchResults(max_results=2)
            for q in state["search_queries"]:
                response = search.invoke(q)     
                for r in response:
                    if 'content' in r:
                        content.append(r['content'])     

        chat = get_chat()
        reflect = reflection_prompt | chat
            
        messages = state["messages"]
        cls_map = {"ai": HumanMessage, "human": AIMessage}
        translated = [messages[0]] + [
            cls_map[msg.type](content=msg.content) for msg in messages[1:]
        ]
        print('translated: ', translated)     
           
        res = reflect.invoke(
            {
                "messages": translated,
                "reflection": state["reflection"],
                "content": content
            }
        )    
                                
        response = HumanMessage(content=res.content[res.content.find('<result>')+8:len(res.content)-9])
        print('response: ', response)
                
        revision_number = state["revision_number"] if state.get("revision_number") is not None else 1
        return {
            "messages": [response], 
            "revision_number": revision_number + 1
        }
    
    MAX_REVISIONS = 1
    def should_continue(state: State, config):
        print("###### should_continue ######")
        max_revisions = config.get("configurable", {}).get("max_revisions", MAX_REVISIONS)
        print("max_revisions: ", max_revisions)
            
        if state["revision_number"] > max_revisions:
            return "end"
        return "continue"

    def buildKnowledgeGuru():    
        workflow = StateGraph(State)

        workflow.add_node("generate", generate)
        workflow.add_node("reflect", reflect)
        workflow.add_node("revise_answer", revise_answer)

        workflow.set_entry_point("generate")

        workflow.add_conditional_edges(
            "revise_answer", 
            should_continue, 
            {
                "end": END, 
                "continue": "reflect"}
        )

        workflow.add_edge("generate", "reflect")
        workflow.add_edge("reflect", "revise_answer")
        
        app = workflow.compile()
        
        return app
    
    app = buildKnowledgeGuru()
        
    isTyping(connectionId, requestId, "")    
    inputs = [HumanMessage(content=query)]
    config = {
        "recursion_limit": 50,
        "max_revisions": MAX_REVISIONS,
        "requestId": requestId,
        "connectionId": connectionId
    }
    
    for output in app.stream({"messages": inputs}, config):   
        for key, value in output.items():
            print(f"Finished: {key}")
            #print("value: ", value)
            
    print('value: ', value)
        
    readStreamMsg(connectionId, requestId, value["messages"][-1].content)
    
    return value["messages"][-1].content

    
####################### LangGraph #######################
# Long form Writing Agent
#########################################################
def run_long_form_writing_agent(connectionId, requestId, query):
    # Workflow - Reflection
    class ReflectionState(TypedDict):
        draft : str
        reflection : List[str]
        search_queries : List[str]
        revised_draft: str
        revision_number: int
        reference: List[str]
        
    class Reflection(BaseModel):
        missing: str = Field(description="Critique of what is missing.")
        advisable: str = Field(description="Critique of what is helpful for better writing")
        superfluous: str = Field(description="Critique of what is superfluous")

    class Research(BaseModel):
        """Provide reflection and then follow up with search queries to improve the writing."""

        reflection: Reflection = Field(description="Your reflection on the initial writing.")
        search_queries: list[str] = Field(
            description="1-3 search queries for researching improvements to address the critique of your current writing."
        )

    class ReflectionKor(BaseModel):
        missing: str = Field(description="작성된 글에 있어야하는데 빠진 내용이나 단점")
        advisable: str = Field(description="더 좋은 글이 되기 위해 추가하여야 할 내용")
        superfluous: str = Field(description="글의 길이나 스타일에 대한 비평")

    class ResearchKor(BaseModel):
        """글쓰기를 개선하기 위한 검색 쿼리를 제공합니다."""

        reflection: ReflectionKor = Field(description="작성된 글에 대한 평가")
        search_queries: list[str] = Field(
            description="현재 글과 관련된 3개 이내의 검색어"
        )    
        
    def reflect_node(state: ReflectionState, config):
        print("###### reflect ######")
        draft = state['draft']
        print('draft: ', draft)
        
        idx = config.get("configurable", {}).get("idx")
        print('reflect_node idx: ', idx)
        update_state_message(f"reflecting... (search_queries-{idx})", config)
    
        reflection = []
        search_queries = []
        for attempt in range(5):
            chat = get_chat()
            if isKorean(draft):
                structured_llm = chat.with_structured_output(ResearchKor, include_raw=True)
            else:
                structured_llm = chat.with_structured_output(Research, include_raw=True)
                
            info = structured_llm.invoke(draft)
            print(f'attempt: {attempt}, info: {info}')
                
            if not info['parsed'] == None:
                parsed_info = info['parsed']
                # print('reflection: ', parsed_info.reflection)                
                reflection = [parsed_info.reflection.missing, parsed_info.reflection.advisable]
                search_queries = parsed_info.search_queries
                
                print('reflection: ', parsed_info.reflection)            
                print('search_queries: ', search_queries)     
        
                if isKorean(draft):
                    translated_search = []
                    for q in search_queries:
                        chat = get_chat()
                        if isKorean(q):
                            search = traslation(chat, q, "Korean", "English")
                        else:
                            search = traslation(chat, q, "English", "Korean")
                        translated_search.append(search)
                        
                    print('translated_search: ', translated_search)
                    search_queries += translated_search

                print('search_queries (mixed): ', search_queries)
                break
        
        revision_number = state["revision_number"] if state.get("revision_number") is not None else 1
        return {
            "reflection": reflection,
            "search_queries": search_queries,
            "revision_number": revision_number + 1
        }

    def retrieve_for_writing(conn, q, config):
        top_k = numberOfDocs
        
        relevant_docs = []
        # RAG - knowledge base
        #if rag_state=='enable':
        #    update_state_message(f"reflecting... (RAG_retriever-{idx})", config)
        #    docs = retrieve_from_knowledge_base(q, top_k)
        #    print(f'q: {q}, RAG: {docs}')
                            
        #    if len(docs):
        #        update_state_message(f"reflecting... (grader-{idx})", config)        
        #        fitered_docs = grade_documents(q, docs)
                
        #        print(f'retrieve {idx}: len(RAG_relevant_docs)=', len(relevant_docs))
        #        relevant_docs += fitered_docs
            
        # web search
        idx = config.get("configurable", {}).get("idx") 
        update_state_message(f"reflecting... (WEB_retriever-{idx})", config)    
        docs = tavily_search(q, top_k)
        print(f'q: {q}, WEB: {docs}')
                
        if len(docs):
            update_state_message(f"reflecting... (grader-{idx})", config)        
            fitered_docs = grade_documents(q, docs)
            
            print(f'retrieve {idx}: len(WEB_relevant_docs)=', len(relevant_docs))
            relevant_docs += fitered_docs
                    
        conn.send(relevant_docs)
        conn.close()

    def parallel_retriever(search_queries, config):
        relevant_documents = []    
        
        processes = []
        parent_connections = []
        for q in search_queries:
            parent_conn, child_conn = Pipe()
            parent_connections.append(parent_conn)
                
            process = Process(target=retrieve_for_writing, args=(child_conn, q, config))
            processes.append(process)

        for process in processes:
            process.start()
                
        for parent_conn in parent_connections:
            rel_docs = parent_conn.recv()

            if(len(rel_docs)>=1):
                for doc in rel_docs:
                    relevant_documents.append(doc)    

        for process in processes:
            process.join()
        
        #print('relevant_docs: ', relevant_docs)
        return relevant_documents

    def retrieve_docs(search_queries, config):
        relevant_docs = []
        top_k = numberOfDocs
        
        idx = config.get("configurable", {}).get("idx")
        
        if multi_region == 'enable':
            relevant_docs = parallel_retriever(search_queries, config)        
        else:
            for q in search_queries:        
                # RAG - knowledge base
                #if rag_state=='enable':
                #    update_state_message(f"reflecting... (RAG_retriever-{idx})", config)
                #    docs = retrieve_from_knowledge_base(q, top_k)
                #    print(f'q: {q}, RAG: {docs}')
                            
                #    if len(docs):
                #        update_state_message(f"reflecting... (grader-{idx})", config)        
                #        relevant_docs += grade_documents(q, docs)
            
                # web search
                update_state_message(f"reflecting... (WEB_retriever-{idx})", config)
                docs = tavily_search(q, top_k)
                print(f'q: {q}, WEB: {docs}')
                
                if len(docs):
                    update_state_message(f"reflecting... (grader-{idx})", config)        
                    relevant_docs += grade_documents(q, docs)
                    
        for i, doc in enumerate(relevant_docs):
            print(f"#### {i}: {doc.page_content[:100]}")
        
        return relevant_docs
        
    def revise_draft(state: ReflectionState, config):   
        print("###### revise_answer ######")
        
        draft = state['draft']
        search_queries = state['search_queries']
        reflection = state['reflection']
        print('draft: ', draft)
        print('search_queries: ', search_queries)
        print('reflection: ', reflection)
                            
        # web search
        idx = config.get("configurable", {}).get("idx")
        print('revise_draft idx: ', idx)
        update_state_message(f"revising... (retrieve-{idx})", config)
        
        filtered_docs = retrieve_docs(search_queries, config)        
        print('filtered_docs: ', filtered_docs)
        
        if 'reference' in state:
            reference = state['reference']
            reference += filtered_docs
        else:
            reference = filtered_docs
        print('len(reference): ', reference)
        
        content = []   
        if len(filtered_docs):
            for d in filtered_docs:
                content.append(d.page_content)            
        print('content: ', content)
        
        update_state_message(f"revising... (generate-{idx})", config)
        
        if isKorean(draft):
            revise_template = (
                "당신은 장문 작성에 능숙한 유능한 글쓰기 도우미입니다."                
                "draft을 critique과 information 사용하여 수정하십시오."
                "최종 결과는 한국어로 작성하고 <result> tag를 붙여주세요."
                            
                "<draft>"
                "{draft}"
                "</draft>"
                            
                "<critique>"
                "{reflection}"
                "</critique>"

                "<information>"
                "{content}"
                "</information>"
            )
        else:    
            revise_template = (
                "You are an excellent writing assistant." 
                "Revise this draft using the critique and additional information."
                # "Provide the final answer using Korean with <result> tag."
                "Provide the final answer with <result> tag."
                            
                "<draft>"
                "{draft}"
                "</draft>"
                            
                "<critique>"
                "{reflection}"
                "</critique>"

                "<information>"
                "{content}"
                "</information>"
            )
                    
        revise_prompt = ChatPromptTemplate([
            ('human', revise_template)
        ])

        chat = get_chat()
        reflect = revise_prompt | chat
           
        res = reflect.invoke(
            {
                "draft": draft,
                "reflection": reflection,
                "content": content
            }
        )
        output = res.content
        # print('output: ', output)
        
        revised_draft = output[output.find('<result>')+8:len(output)-9]
        # print('revised_draft: ', revised_draft) 
            
        if revised_draft.find('#')!=-1 and revised_draft.find('#')!=0:
            revised_draft = revised_draft[revised_draft.find('#'):]

        print('--> draft: ', draft)
        print('--> reflection: ', reflection)
        print('--> revised_draft: ', revised_draft)
        
        revision_number = state["revision_number"] if state.get("revision_number") is not None else 1
        
        return {
            "revised_draft": revised_draft,            
            "revision_number": revision_number,
            "reference": reference
        }
        
    MAX_REVISIONS = 1
    def should_continue(state: ReflectionState, config):
        print("###### should_continue ######")
        max_revisions = config.get("configurable", {}).get("max_revisions", MAX_REVISIONS)
        print("max_revisions: ", max_revisions)
            
        if state["revision_number"] > max_revisions:
            return "end"
        return "continue"        
    
    def buildReflection():
        workflow = StateGraph(ReflectionState)

        # Add nodes
        workflow.add_node("reflect_node", reflect_node)
        workflow.add_node("revise_draft", revise_draft)

        # Set entry point
        workflow.set_entry_point("reflect_node")
        
        workflow.add_conditional_edges(
            "revise_draft", 
            should_continue, 
            {
                "end": END, 
                "continue": "reflect_node"
            }
        )

        # Add edges
        workflow.add_edge("reflect_node", "revise_draft")
        
        return workflow.compile()
    
    # Workflow - Long Writing
    class State(TypedDict):
        instruction : str
        planning_steps : List[str]
        drafts : List[str]
        # num_steps : int
        final_doc : str
        word_count : int
            
    def plan_node(state: State, config):
        print("###### plan ######")
        instruction = state["instruction"]
        print('subject: ', instruction)
        
        update_state_message("planning...", config)
        
        if isKorean(instruction):
            planner_template = (
                "당신은 장문 작성에 능숙한 유능한 글쓰기 도우미입니다."
                "이번 글쓰기는 20,000 단어 이상의 장편을 목표로 합니다."
                "당신은 글쓰기 지시 사항을 여러 개의 하위 작업으로 나눌 것입니다."
                "각 하위 작업은 에세이의 한 단락 작성을 안내할 것이며, 해당 단락의 주요 내용과 단어 수 요구 사항을 포함해야 합니다."

                "글쓰기 지시 사항:"
                "<instruction>"
                "{instruction}"
                "<instruction>"
                
                "다음 형식으로 나누어 주시기 바랍니다. 각 하위 작업은 한 줄을 차지합니다:"
                "1. Main Point: [문단의 주요 내용을 자세히 설명하십시오.], Word Count: [Word count requirement, e.g., 800 words]"
                "2. Main Point: [문단의 주요 내용을 자세히 설명하십시오.], Word Count: [word count requirement, e.g. 1500 words]."
                "..."
                
                "각 하위 작업이 명확하고 구체적인지, 그리고 모든 하위 작업이 작문 지시 사항의 전체 내용을 다루고 있는지 확인하세요."
                "과제를 너무 세분화하지 마세요. 각 하위 과제의 문단은 500단어 이상 3000단어 이하여야 합니다."
                "다른 내용은 출력하지 마십시오. 이것은 진행 중인 작업이므로 열린 결론이나 다른 수사학적 표현을 생략하십시오."                
            )
        else:
            planner_template = (
                "You are a helpful assistant highly skilled in long-form writing."
                "This writing aims for a novel of over 20,000 words."
                "You will break down the writing instruction into multiple subtasks."
                "Each subtask will guide the writing of one paragraph in the essay, and should include the main points and word count requirements for that paragraph."

                "The writing instruction is as follows:"
                "<instruction>"
                "{instruction}"
                "<instruction>"
                
                "Please break it down in the following format, with each subtask taking up one line:"
                "1. Main Point: [Describe the main point of the paragraph, in detail], Word Count: [Word count requirement, e.g., 800 words]"
                "2. Main Point: [Describe the main point of the paragraph, in detail], Word Count: [word count requirement, e.g. 1500 words]."
                "..."
                
                "Make sure that each subtask is clear and specific, and that all subtasks cover the entire content of the writing instruction."
                "Do not split the subtasks too finely; each subtask's paragraph should be no less than 500 words and no more than 3000 words."
                "Do not output any other content. As this is an ongoing work, omit open-ended conclusions or other rhetorical hooks."                
            )
        
        planner_prompt = ChatPromptTemplate([
            ('human', planner_template) 
        ])
                
        chat = get_chat()
        
        planner = planner_prompt | chat
    
        response = planner.invoke({"instruction": instruction})
        print('response: ', response.content)
    
        plan = response.content.strip().replace('\n\n', '\n')
        planning_steps = plan.split('\n')        
        print('planning_steps: ', planning_steps)
            
        return {
            "instruction": instruction,
            "planning_steps": planning_steps
        }
        
    def execute_node(state: State, config):
        print("###### write (execute) ######")        
        instruction = state["instruction"]
        planning_steps = state["planning_steps"]
        print('instruction: ', instruction)
        print('planning_steps: ', planning_steps)
        
        update_state_message("executing...", config)
        
        if isKorean(instruction):
            write_template = (
                "당신은 훌륭한 글쓰기 도우미입니다." 
                "아래와 같이 원본 글쓰기 지시사항과 계획한 글쓰기 단계를 제공하겠습니다."
                "또한 제가 이미 작성한 텍스트를 제공합니다."

                "글쓰기 지시사항:"
                "<instruction>"
                "{intructions}"
                "</instruction>"

                "글쓰기 단계:"
                "<plan>"
                "{plan}"
                "</plan>"

                "이미 작성한 텍스트:"
                "<text>"
                "{text}"
                "</text>"

                "글쓰기 지시 사항, 글쓰기 단계, 이미 작성된 텍스트를 참조하여 다음 단계을 계속 작성합니다."
                "다음 단계:"
                "<step>"
                "{STEP}"
                "</step>"
                
                "글이 끊어지지 않고 잘 이해되도록 하나의 문단을 충분히 길게 작성합니다."
                "필요하다면 앞에 작은 부제를 추가할 수 있습니다."
                "이미 작성된 텍스트를 반복하지 말고 작성한 문단만 출력하세요."                
                "Markdown 포맷으로 서식을 작성하세요."
                "최종 결과에 <result> tag를 붙여주세요."
            )
        else:    
            write_template = (
                "You are an excellent writing assistant." 
                "I will give you an original writing instruction and my planned writing steps."
                "I will also provide you with the text I have already written."
                "Please help me continue writing the next paragraph based on the writing instruction, writing steps, and the already written text."

                "Writing instruction:"
                "<instruction>"
                "{intructions}"
                "</instruction>"

                "Writing steps:"
                "<plan>"
                "{plan}"
                "</plan>"

                "Already written text:"
                "<text>"
                "{text}"
                "</text>"

                "Please integrate the original writing instruction, writing steps, and the already written text, and now continue writing {STEP}."
                "If needed, you can add a small subtitle at the beginning."
                "Remember to only output the paragraph you write, without repeating the already written text."
                
                "Use markdown syntax to format your output:"
                "- Headings: # for main, ## for sections, ### for subsections, etc."
                "- Lists: * or - for bulleted, 1. 2. 3. for numbered"
                "- Do not repeat yourself"
                "Provide the final answer with <result> tag."
                #"Provide the final answer using Korean with <result> tag."
            )

        write_prompt = ChatPromptTemplate([
            ('human', write_template)
        ])
        
        text = ""
        drafts = []
        if len(planning_steps) > 50:
            print("plan is too long")
            # print(plan)
            return
        
        for idx, step in enumerate(planning_steps):
            update_state_message(f"executing... (step: {idx+1}/{len(planning_steps)})", config)
            
            # Invoke the write_chain
            chat = get_chat()
            write_chain = write_prompt | chat
            
            result = write_chain.invoke({
                "intructions": instruction,
                "plan": planning_steps,
                "text": text,
                "STEP": step
            })            
            output = result.content
            # print('output: ', output)
            
            draft = output[output.find('<result>')+8:len(output)-9]
            # print('draft: ', draft) 
                       
            if draft.find('#')!=-1 and draft.find('#')!=0:
                draft = draft[draft.find('#'):]
            
            print(f"--> step:{step}")
            print(f"--> {draft}")
                
            drafts.append(draft)
            text += draft + '\n\n'

        return {
            "instruction": instruction,
            "drafts": drafts
        }

    def reflect_draft(conn, reflection_app, idx, config, draft):     
        inputs = {
            "draft": draft
        }            
        output = reflection_app.invoke(inputs, config)
        
        print('idx: ', idx)
        
        result = {
            "revised_draft": output['revised_draft'],
            "idx": idx,
            "reference": output['reference']
        }
            
        conn.send(result)    
        conn.close()
        
    def reflect_drafts_using_parallel_processing(drafts, config):
        revised_drafts = drafts
        
        processes = []
        parent_connections = []
        references = []
        
        reflection_app = buildReflection()
        
        requestId = config.get("configurable", {}).get("requestId", "")
        print('requestId: ', requestId)
        connectionId = config.get("configurable", {}).get("connectionId", "")
        print('connectionId: ', connectionId)
        
        for idx, draft in enumerate(drafts):
            parent_conn, child_conn = Pipe()
            parent_connections.append(parent_conn)
            
            print(f"idx:{idx} --> draft:{draft}")
            
            app_config = {
                "recursion_limit": 50,
                "max_revisions": MAX_REVISIONS,
                "requestId":requestId,
                "connectionId": connectionId,
                "idx": idx
            }
            process = Process(target=reflect_draft, args=(child_conn, reflection_app, idx, app_config, draft))
            processes.append(process)
            
        for process in processes:
            process.start()
                
        for parent_conn in parent_connections:
            result = parent_conn.recv()

            if result is not None:
                print('result: ', result)
                revised_drafts[result['idx']] = result['revised_draft']
                references += result['reference']

        for process in processes:
            process.join()
                
        final_doc = ""   
        for revised_draft in revised_drafts:
            final_doc += revised_draft + '\n\n'
        
        return final_doc, references

    def get_subject(query):
        system = (
            "Extract the subject of the question in 6 words or fewer."
        )
        
        human = "<question>{question}</question>"
        
        prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
        # print('prompt: ', prompt)
        
        chat = get_chat()
        chain = prompt | chat    
        try: 
            result = chain.invoke(
                {
                    "question": query
                }
            )        
            subject = result.content
            # print('the subject of query: ', subject)
            
        except Exception:
            err_msg = traceback.format_exc()
            print('error message: ', err_msg)                    
            raise Exception ("Not able to request to LLM")        
        return subject
    
    def markdown_to_html(body, reference):
        body = body + f"\n\n### 참고자료\n\n\n"
        
        html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta http-equiv="X-UA-Compatible" content="IE=edge">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        
        <md-block>
        </md-block>
        <script type="module" src="https://md-block.verou.me/md-block.js"></script>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.2.0/github-markdown-light.css" integrity="sha512-n5zPz6LZB0QV1eraRj4OOxRbsV7a12eAGfFcrJ4bBFxxAwwYDp542z5M0w24tKPEhKk2QzjjIpR5hpOjJtGGoA==" crossorigin="anonymous" referrerpolicy="no-referrer"/>
    </head>
    <body>
        <div class="markdown-body">
            <md-block>{body}
            </md-block>
        </div>
        {reference}
    </body>
    </html>"""        
        return html

    def get_references_for_html(docs):
        reference = ""
        nameList = []
        cnt = 1
        for i, doc in enumerate(docs):
            print(f"reference {i}: doc")
            page = ""
            if "page" in doc.metadata:
                page = doc.metadata['page']
                #print('page: ', page)            
            url = ""
            if "url" in doc.metadata:
                url = doc.metadata['url']
                #print('url: ', url)                
            name = ""
            if "name" in doc.metadata:
                name = doc.metadata['name']
                #print('name: ', name)     
            pos = name.rfind('/')
            name = name[pos+1:]
            print(f"name: {name}")
            
            excerpt = ""+doc.page_content

            excerpt = re.sub('"', '', excerpt)
            print('length: ', len(excerpt))
            
            if name in nameList:
                print('duplicated!')
            else:
                reference = reference + f"{cnt}. <a href={url} target=_blank>{name}</a><br>"
                nameList.append(name)
                cnt = cnt+1
                
        return reference

    def revise_answer(state: State, config):
        print("###### revise ######")
        drafts = state["drafts"]        
        print('drafts: ', drafts)
        
        update_state_message("revising...", config)
        
        # reflection
        if multi_region == 'enable':  # parallel processing
            final_doc, references = reflect_drafts_using_parallel_processing(drafts, config)
        else:
            reflection_app = buildReflection()
                
            final_doc = ""   
            references = []
            
            requestId = config.get("configurable", {}).get("requestId", "")
            print('requestId: ', requestId)
            connectionId = config.get("configurable", {}).get("connectionId", "")
            print('connectionId: ', connectionId)
            
            for idx, draft in enumerate(drafts):
                inputs = {
                    "draft": draft
                }                    
                app_config = {
                    "recursion_limit": 50,
                    "max_revisions": MAX_REVISIONS,
                    "requestId":requestId,
                    "connectionId": connectionId,
                    "idx": idx
                }
                output = reflection_app.invoke(inputs, config=app_config)
                final_doc += output['revised_draft'] + '\n\n'
                references += output['reference']

        subject = get_subject(state['instruction'])
        subject = subject.replace(" ","_")
        subject = subject.replace("?","")
        subject = subject.replace("!","")
        subject = subject.replace(".","")
        subject = subject.replace(":","")
        
        print('len(references): ', len(references))
        
        # markdown file
        markdown_key = 'markdown/'+f"{subject}.md"
        # print('markdown_key: ', markdown_key)
        
        markdown_body = f"## {state['instruction']}\n\n"+final_doc
                
        s3_client = boto3.client('s3')  
        response = s3_client.put_object(
            Bucket=s3_bucket,
            Key=markdown_key,
            ContentType='text/markdown',
            Body=markdown_body.encode('utf-8')
        )
        # print('response: ', response)
        
        markdown_url = f"{path}{markdown_key}"
        print('markdown_url: ', markdown_url)
        
        # html file
        html_key = 'markdown/'+f"{subject}.html"
        
        html_reference = ""
        print('references: ', references)
        if references:
            html_reference = get_references_for_html(references)
            
            global reference_docs
            reference_docs += references
            
        html_body = markdown_to_html(markdown_body, html_reference)
        print('html_body: ', html_body)
        
        s3_client = boto3.client('s3')  
        response = s3_client.put_object(
            Bucket=s3_bucket,
            Key=html_key,
            ContentType='text/html',
            Body=html_body
        )
        # print('response: ', response)
        
        html_url = f"{path}{html_key}"
        print('html_url: ', html_url)
        
        return {
            "final_doc": final_doc+f"\n<a href={html_url} target=_blank>[미리보기 링크]</a>\n<a href={markdown_url} download=\"{subject}.md\">[다운로드 링크]</a>"
        }
        
    def buildLongformWriting():
        workflow = StateGraph(State)

        # Add nodes
        workflow.add_node("planning_node", plan_node)
        workflow.add_node("execute_node", execute_node)
        workflow.add_node("revising_node", revise_answer)

        # Set entry point
        workflow.set_entry_point("planning_node")

        # Add edges
        workflow.add_edge("planning_node", "execute_node")
        workflow.add_edge("execute_node", "revising_node")
        workflow.add_edge("revising_node", END)
        
        return workflow.compile()
    
    app = buildLongformWriting()
    
    # Run the workflow
    isTyping(connectionId, requestId, "")        
    inputs = {
        "instruction": query
    }    
    config = {
        "recursion_limit": 50,
        "requestId": requestId,
        "connectionId": connectionId
    }
    
    output = app.invoke(inputs, config)
    print('output: ', output)
    
    return output['final_doc']
            
####################### Knowledge Base #######################
# Knowledge Base
##############################################################

def query_using_RAG_context(connectionId, requestId, chat, context, revised_question):    
    isTyping(connectionId, requestId, "generating...")  
                
    if isKorean(revised_question)==True:
        system = (
            """다음의 <context> tag안의 참고자료를 이용하여 상황에 맞는 구체적인 세부 정보를 충분히 제공합니다. Assistant의 이름은 서연이고, 모르는 질문을 받으면 솔직히 모른다고 말합니다.
            
            <context>
            {context}
            </context>"""
        )
    else: 
        system = (
            """Here is pieces of context, contained in <context> tags. Provide a concise answer to the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
            
            <context>
            {context}
            </context>"""
        )
    
    human = "{input}"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    # print('prompt: ', prompt)
                   
    chain = prompt | chat
    
    try: 
        stream = chain.invoke(
            {
                "context": context,
                "input": revised_question,
            }
        )
        msg = readStreamMsg(connectionId, requestId, stream.content)    
        print('msg: ', msg)
        
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)        
            
        sendErrorMessage(connectionId, requestId, err_msg)    
        raise Exception ("Not able to request to LLM")

    return msg

def get_reference_of_knoweledge_base(docs, path, doc_prefix):
    reference = "\n\nFrom\n"
    #print('path: ', path)
    #print('doc_prefix: ', doc_prefix)
    #print('prefix: ', f"/{doc_prefix}")
    
    for i, document in enumerate(docs):
        if document.page_content:
            excerpt = document.page_content
        
        score = document.metadata["score"]
        print('score:', score)
        doc_prefix = "knowledge-base"
        
        link = ""
        if "s3Location" in document.metadata["location"]:
            link = document.metadata["location"]["s3Location"]["uri"] if document.metadata["location"]["s3Location"]["uri"] is not None else ""
            
            print('link:', link)    
            pos = link.find(f"/{doc_prefix}")
            name = link[pos+len(doc_prefix)+1:]
            encoded_name = parse.quote(name)
            print('name:', name)
            link = f"{path}{doc_prefix}{encoded_name}"
            
        elif "webLocation" in document.metadata["location"]:
            link = document.metadata["location"]["webLocation"]["url"] if document.metadata["location"]["webLocation"]["url"] is not None else ""
            name = "WWW"

        print('link:', link)
                    
        reference = reference + f"{i+1}. <a href={link} target=_blank>{name}</a>, <a href=\"#\" onClick=\"alert(`{excerpt}`)\">관련문서</a>\n"
                    
    return reference


        
#########################################################
def traslation(chat, text, input_language, output_language):
    system = (
        "You are a helpful assistant that translates {input_language} to {output_language} in <article> tags. Put it in <result> tags."
    )
    human = "<article>{text}</article>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    # print('prompt: ', prompt)
    
    chain = prompt | chat    
    try: 
        result = chain.invoke(
            {
                "input_language": input_language,
                "output_language": output_language,
                "text": text,
            }
        )
        
        msg = result.content
        # print('translated text: ', msg)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                    
        raise Exception ("Not able to request to LLM")

    return msg[msg.find('<result>')+8:len(msg)-9] # remove <result> tag

def summary_of_code(chat, code, mode):
    if mode == 'py':
        system = (
            "다음의 <article> tag에는 python code가 있습니다. code의 전반적인 목적에 대해 설명하고, 각 함수의 기능과 역할을 자세하게 한국어 500자 이내로 설명하세요."
        )
    elif mode == 'js':
        system = (
            "다음의 <article> tag에는 node.js code가 있습니다. code의 전반적인 목적에 대해 설명하고, 각 함수의 기능과 역할을 자세하게 한국어 500자 이내로 설명하세요."
        )
    else:
        system = (
            "다음의 <article> tag에는 code가 있습니다. code의 전반적인 목적에 대해 설명하고, 각 함수의 기능과 역할을 자세하게 한국어 500자 이내로 설명하세요."
        )
    
    human = "<article>{code}</article>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    # print('prompt: ', prompt)
    
    chain = prompt | chat    
    try: 
        result = chain.invoke(
            {
                "code": code
            }
        )
        
        summary = result.content
        print('result of code summarization: ', summary)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                    
        raise Exception ("Not able to request to LLM")
    
    return summary

def revise_question(connectionId, requestId, chat, query):    
    print("###### revise_question ######")
    
    global history_length, token_counter_history    
    history_length = token_counter_history = 0
    
    isTyping(connectionId, requestId, "revising...")
        
    if isKorean(query)==True :      
        system = (
            ""
        )  
        human = """이전 대화를 참조하여, 다음의 <question>의 뜻을 명확히 하는 새로운 질문을 한국어로 생성하세요. 새로운 질문은 원래 질문의 중요한 단어를 반드시 포함합니다. 결과는 <result> tag를 붙여주세요.
        
        <question>            
        {question}
        </question>"""
        
    else: 
        system = (
            ""
        )
        human = """Rephrase the follow up <question> to be a standalone question. Put it in <result> tags.
        <question>            
        {question}
        </question>"""
            
    prompt = ChatPromptTemplate.from_messages([("system", system), MessagesPlaceholder(variable_name="history"), ("human", human)])
    # print('prompt: ', prompt)
    
    history = memory_chain.load_memory_variables({})["chat_history"]
    # print('memory_chain: ', history)
                
    chain = prompt | chat    
    try: 
        result = chain.invoke(
            {
                "history": history,
                "question": query,
            }
        )
        generated_question = result.content
        
        revised_question = generated_question[generated_question.find('<result>')+8:len(generated_question)-9] # remove <result> tag                   
        # print('revised_question: ', revised_question)
        
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)        
            
        sendErrorMessage(connectionId, requestId, err_msg)    
        raise Exception ("Not able to request to LLM")

    if debugMessageMode == 'true':  
        chat_history = ""
        for dialogue_turn in history:
            #print('type: ', dialogue_turn.type)
            #print('content: ', dialogue_turn.content)
            
            dialog = f"{dialogue_turn.type}: {dialogue_turn.content}\n"            
            chat_history = chat_history + dialog
                
        history_length = len(chat_history)
        print('chat_history length: ', history_length)
        
        token_counter_history = 0
        if chat_history:
            token_counter_history = chat.get_num_tokens(chat_history)
            print('token_size of history: ', token_counter_history)
            
        sendDebugMessage(connectionId, requestId, f"새로운 질문: {revised_question}\n * 대화이력({str(history_length)}자, {token_counter_history} Tokens)을 활용하였습니다.")
            
    return revised_question    
    # return revised_question.replace("\n"," ")

def isTyping(connectionId, requestId, msg):    
    if not msg:
        msg = "typing a message..."
    msg_proceeding = {
        'request_id': requestId,
        'msg': msg,
        'status': 'istyping'
    }
    #print('result: ', json.dumps(result))
    sendMessage(connectionId, msg_proceeding)

def removeFunctionXML(msg):
    #print('msg: ', msg)
    
    while(1):
        start_index = msg.find('<function_calls>')
        end_index = msg.find('</function_calls>')
        length = 18
        
        if start_index == -1:
            start_index = msg.find('<invoke>')
            end_index = msg.find('</invoke>')
            length = 10
        
        output = ""
        if start_index>=0:
            # print('start_index: ', start_index)
            # print('msg: ', msg)
            
            if start_index>=1:
                output = msg[:start_index-1]
                
                if output == "\n" or output == "\n\n":
                    output = ""
            
            if end_index >= 1:
                # print('end_index: ', end_index)
                output = output + msg[end_index+length:]
                            
            msg = output
        else:
            output = msg
            break

    return output

def readStreamMsgForAgent(connectionId, requestId, stream):
    msg = ""
    if stream:
        for event in stream:
            #print('event: ', event)
            msg = msg + event
            
            output = removeFunctionXML(msg)
            # print('output: ', output)
            
            if len(output)>0 and output[0]!='<':
                result = {
                    'request_id': requestId,
                    'msg': output,
                    'status': 'proceeding'
                }
                #print('result: ', json.dumps(result))
                sendMessage(connectionId, result)
            
def readStreamMsg(connectionId, requestId, stream):
    msg = ""
    if stream:
        for event in stream:
            #print('event: ', event)
            msg = msg + event
            
            result = {
                'request_id': requestId,
                'msg': msg,
                'status': 'proceeding'
            }
            #print('result: ', json.dumps(result))
            sendMessage(connectionId, result)
    # print('msg: ', msg)
    return msg
    
def sendMessage(id, body):
    # print('sendMessage size: ', len(body))
    try:
        client.post_to_connection(
            ConnectionId=id, 
            Data=json.dumps(body)
        )
    except Exception:
        err_msg = traceback.format_exc()
        print('err_msg: ', err_msg)
        # raise Exception ("Not able to send a message")

def sendResultMessage(connectionId, requestId, msg):    
    result = {
        'request_id': requestId,
        'msg': msg,
        'status': 'completed'
    }
    #print('debug: ', json.dumps(debugMsg))
    sendMessage(connectionId, result)
    
def sendDebugMessage(connectionId, requestId, msg):
    debugMsg = {
        'request_id': requestId,
        'msg': msg,
        'status': 'debug'
    }
    #print('debug: ', json.dumps(debugMsg))
    sendMessage(connectionId, debugMsg)    
        
def sendErrorMessage(connectionId, requestId, msg):
    errorMsg = {
        'request_id': requestId,
        'msg': msg,
        'status': 'error'
    }
    print('error: ', json.dumps(errorMsg))
    sendMessage(connectionId, errorMsg)    

def load_chat_history(userId, allowTime):
    dynamodb_client = boto3.client('dynamodb')

    response = dynamodb_client.query(
        TableName=callLogTableName,
        KeyConditionExpression='user_id = :userId AND request_time > :allowTime',
        ExpressionAttributeValues={
            ':userId': {'S': userId},
            ':allowTime': {'S': allowTime}
        }
    )
    # print('query result: ', response['Items'])

    for item in response['Items']:
        text = item['body']['S']
        msg = item['msg']['S']
        type = item['type']['S']

        if type == 'text':
            memory_chain.chat_memory.add_user_message(text)
            if len(msg) > MSG_LENGTH:
                memory_chain.chat_memory.add_ai_message(msg[:MSG_LENGTH])                          
            else:
                memory_chain.chat_memory.add_ai_message(msg)     

def translate_text(chat, text):
    system = (
        "You are a helpful assistant that translates {input_language} to {output_language} in <article> tags. Put it in <result> tags."
    )
    human = "<article>{text}</article>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    # print('prompt: ', prompt)
    
    if isKorean(text)==False :
        input_language = "English"
        output_language = "Korean"
    else:
        input_language = "Korean"
        output_language = "English"
                        
    chain = prompt | chat    
    try: 
        result = chain.invoke(
            {
                "input_language": input_language,
                "output_language": output_language,
                "text": text,
            }
        )
        msg = result.content
        print('translated text: ', msg)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                    
        raise Exception ("Not able to request to LLM")

    return msg[msg.find('<result>')+8:len(msg)-9] # remove <result> tag

def check_grammer(chat, text):
    if isKorean(text)==True:
        system = (
            "다음의 <article> tag안의 문장의 오류를 찾아서 설명하고, 오류가 수정된 문장을 답변 마지막에 추가하여 주세요."
        )
    else: 
        system = (
            "Here is pieces of article, contained in <article> tags. Find the error in the sentence and explain it, and add the corrected sentence at the end of your answer."
        )
        
    human = "<article>{text}</article>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    # print('prompt: ', prompt)
    
    chain = prompt | chat    
    try: 
        result = chain.invoke(
            {
                "text": text
            }
        )
        
        msg = result.content
        print('result of grammer correction: ', msg)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                    
        raise Exception ("Not able to request to LLM")
    
    return msg

def use_multimodal(img_base64, query):    
    multimodal = get_multimodal()
    
    if query == "":
        query = "그림에 대해 상세히 설명해줘."
    
    messages = [
        SystemMessage(content="답변은 500자 이내의 한국어로 설명해주세요."),
        HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}", 
                    },
                },
                {
                    "type": "text", "text": query
                },
            ]
        )
    ]
    
    try: 
        result = multimodal.invoke(messages)
        
        summary = result.content
        print('result of code summarization: ', summary)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                    
        raise Exception ("Not able to request to LLM")
    
    return summary

def extract_text(img_base64):    
    multimodal = get_multimodal()
    query = "텍스트를 추출해서 utf8로 변환하세요. <result> tag를 붙여주세요."
    
    messages = [
        HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}", 
                    },
                },
                {
                    "type": "text", "text": query
                },
            ]
        )
    ]
    
    try: 
        result = multimodal.invoke(messages)
        
        extracted_text = result.content
        print('result of text extraction from an image: ', extracted_text)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                    
        raise Exception ("Not able to request to LLM")
    
    return extracted_text

def getResponse(connectionId, jsonBody):
    print('jsonBody: ', jsonBody)
    
    userId  = jsonBody['user_id']
    print('userId: ', userId)
    requestId  = jsonBody['request_id']
    print('requestId: ', requestId)
    requestTime  = jsonBody['request_time']
    print('requestTime: ', requestTime)
    type  = jsonBody['type']
    print('type: ', type)
    body = jsonBody['body']
    print('body: ', body)
    convType = jsonBody['convType']
    print('convType: ', convType)
    
    global multi_region    
    if "multi_region" in jsonBody:
        multi_region = jsonBody['multi_region']
    print('multi_region: ', multi_region)
    
    global grade_state
    if "grade" in jsonBody:
        grade_state = jsonBody['grade']
    else:
        grade_state = 'LLM'
    print('grade_state: ', grade_state)
        
    print('initiate....')
    global reference_docs
    reference_docs = []

    global map_chain, memory_chain
    
    # Multi-LLM
    global selected_chat, length_of_models
    if multi_region == 'enable':
        length_of_models = len(multi_region_models)
        if selected_chat >= length_of_models:
            selected_chat = 0
        profile = multi_region_models[selected_chat]
        
    else:
        length_of_models = len(LLM_for_chat)
        if selected_chat >= length_of_models:
            selected_chat = 0
        profile = LLM_for_chat[selected_chat]    
        
    print('length_of_models: ', length_of_models)
    bedrock_region =  profile['bedrock_region']
    modelId = profile['model_id']
    print(f'selected_chat: {selected_chat}, bedrock_region: {bedrock_region}, modelId: {modelId}')
    
    chat = get_chat()    
    
    # create memory
    if userId in map_chain:  
        print('memory exist. reuse it!')
        memory_chain = map_chain[userId]
    else: 
        print('memory does not exist. create new one!')        
        memory_chain = ConversationBufferWindowMemory(memory_key="chat_history", output_key='answer', return_messages=True, k=5)
        map_chain[userId] = memory_chain

        allowTime = getAllowTime()
        load_chat_history(userId, allowTime)
    
    start = int(time.time())    

    msg = ""
    reference = ""
    
    if type == 'text' and body[:11] == 'list models':
        bedrock_client = boto3.client(
            service_name='bedrock',
            region_name=bedrock_region,
        )
        modelInfo = bedrock_client.list_foundation_models()    
        print('models: ', modelInfo)

        msg = f"The list of models: \n"
        lists = modelInfo['modelSummaries']
        
        for model in lists:
            msg += f"{model['modelId']}\n"
        
        msg += f"current model: {modelId}"
        print('model lists: ', msg)    
        sendResultMessage(connectionId, requestId, msg)
    else:             
        if type == 'text':
            text = body
            print('query: ', text)

            querySize = len(text)
            textCount = len(text.split())
            print(f"query size: {querySize}, words: {textCount}")

            if text == 'clearMemory':
                memory_chain.clear()
                map_chain[userId] = memory_chain
                    
                print('initiate the chat memory!')
                msg  = "The chat memory was intialized in this session."
                sendResultMessage(connectionId, requestId, msg)
            
            elif type == 'text' and body[:21] == 'reflesh current index':
                # reflesh index
                isTyping(connectionId, requestId, "")
                reflesh_opensearch_index()
                msg = "The index was refleshed in OpenSearch."
                sendResultMessage(connectionId, requestId, msg)
                
            else:            
                ########## Basic ##########
                if convType == 'normal':      # normal
                    msg = general_conversation(connectionId, requestId, chat, text)
                elif convType == "translation":
                    msg = translate_text(chat, text) 
                
                elif convType == "grammar":
                    msg = check_grammer(chat, text)  
                    
                ########## RAG ##########
                elif convType == 'rag-opensearch':   # RAG - Vector
                    msg = get_answer_using_opensearch(chat, text, connectionId, requestId)                    

                elif convType == 'rag-opensearch-chat':   # RAG - Vector
                    revised_question = revise_question(connectionId, requestId, chat, text)     
                    print('revised_question: ', revised_question)
                    msg = get_answer_using_opensearch(chat, revised_question, connectionId, requestId)                    
                        
                elif convType == 'agent-crag':  # corrective RAG
                    msg = run_corrective_rag(connectionId, requestId, text)
                                        
                elif convType == 'agent-srag':  # self RAG 
                    msg = run_self_rag(connectionId, requestId, text)
                    
                elif convType == 'agent-scrag':  # self-corrective RAG
                    msg = run_self_corrective_rag(connectionId, requestId, text)        
                    
                
                ########## Agent ##########
                elif convType == 'agent-executor':  # Tool Use
                    msg = run_agent_executor2(connectionId, requestId, text)
                        
                elif convType == 'agent-executor-chat':
                    revised_question = revise_question(connectionId, requestId, chat, text)     
                    print('revised_question: ', revised_question)  
                    msg = run_agent_executor2(connectionId, requestId, revised_question)
                    
                elif convType == 'agent-knowledge-guru':  # Refection: knowledge guru 
                    msg = run_knowledge_guru(connectionId, requestId, text)      
                
                elif convType == 'agent-plan-and-execute':  # plan and execute
                    msg = run_plan_and_exeucute(connectionId, requestId, text)        
                                                                                
                elif convType == 'long-form-writing-agent':  # Multi-agent: long writing
                    msg = run_long_form_writing_agent(connectionId, requestId, text)
                                    
                else:
                    msg = general_conversation(connectionId, requestId, chat, text)  
                    
                memory_chain.chat_memory.add_user_message(text)
                memory_chain.chat_memory.add_ai_message(msg)
                
                if reference_docs:
                    reference = get_references(reference_docs)
                                        
        elif type == 'document':
            isTyping(connectionId, requestId, "")
            
            object = body
            file_type = object[object.rfind('.')+1:len(object)]            
            print('file_type: ', file_type)
            
            if file_type == 'csv':
                docs = load_csv_document(object)
                contexts = []
                for doc in docs:
                    contexts.append(doc.page_content)
                print('contexts: ', contexts)
            
                msg = get_summary(chat, contexts)

            elif file_type == 'pdf' or file_type == 'txt' or file_type == 'md' or file_type == 'pptx' or file_type == 'docx':
                texts = load_document(file_type, object)

                docs = []
                for i in range(len(texts)):
                    docs.append(
                        Document(
                            page_content=texts[i],
                            metadata={
                                'name': object,
                                # 'page':i+1,
                                'url': path+doc_prefix+parse.quote(object)
                            }
                        )
                    )
                print('docs[0]: ', docs[0])    
                print('docs size: ', len(docs))

                contexts = []
                for doc in docs:
                    contexts.append(doc.page_content)
                print('contexts: ', contexts)

                msg = get_summary(chat, contexts)
                
            elif file_type == 'py' or file_type == 'js':
                s3r = boto3.resource("s3")
                doc = s3r.Object(s3_bucket, s3_prefix+'/'+object)
                
                contents = doc.get()['Body'].read().decode('utf-8')
                
                #contents = load_code(file_type, object)                
                                
                msg = summary_of_code(chat, contents, file_type)                  
                
            elif file_type == 'png' or file_type == 'jpeg' or file_type == 'jpg':
                print('multimodal: ', object)
                
                s3_client = boto3.client('s3') 
                    
                image_obj = s3_client.get_object(Bucket=s3_bucket, Key=s3_prefix+'/'+object)
                # print('image_obj: ', image_obj)
                
                image_content = image_obj['Body'].read()
                img = Image.open(BytesIO(image_content))
                
                width, height = img.size 
                print(f"width: {width}, height: {height}, size: {width*height}")
                
                isResized = False
                while(width*height > 5242880):                    
                    width = int(width/2)
                    height = int(height/2)
                    isResized = True
                    print(f"width: {width}, height: {height}, size: {width*height}")
                
                if isResized:
                    img = img.resize((width, height))
                
                buffer = BytesIO()
                img.save(buffer, format="PNG")
                img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                
                command = ""        
                if 'command' in jsonBody:
                    command  = jsonBody['command']
                    print('command: ', command)
                
                # verify the image
                msg = use_multimodal(img_base64, command)       
                
                # extract text from the image
                text = extract_text(img_base64)
                extracted_text = text[text.find('<result>')+8:len(text)-9] # remove <result> tag
                print('extracted_text: ', extracted_text)
                if len(extracted_text)>10:
                    msg = msg + f"\n\n[추출된 Text]\n{extracted_text}\n"
                
                memory_chain.chat_memory.add_user_message(f"{object}에서 텍스트를 추출하세요.")
                memory_chain.chat_memory.add_ai_message(extracted_text)
            
            else:
                msg = "uploaded file: "+object
        
        sendResultMessage(connectionId, requestId, msg+reference)
        # print('msg+reference: ', msg+reference)    
                
        elapsed_time = int(time.time()) - start
        print("total run time(sec): ", elapsed_time)
        
        print('msg: ', msg)

        item = {
            'user_id': {'S':userId},
            'request_id': {'S':requestId},
            'request_time': {'S':requestTime},
            'type': {'S':type},
            'body': {'S':body},
            'msg': {'S':msg+reference}
        }
        client = boto3.client('dynamodb')
        try:
            resp =  client.put_item(TableName=callLogTableName, Item=item)
        except Exception:
            err_msg = traceback.format_exc()
            print('error message: ', err_msg)
            # raise Exception ("Not able to write into dynamodb")         
        #print('resp, ', resp)

    return msg, reference

def lambda_handler(event, context):
    # print('event: ', event)
    
    msg = ""
    if event['requestContext']: 
        connectionId = event['requestContext']['connectionId']        
        routeKey = event['requestContext']['routeKey']
        
        if routeKey == '$connect':
            print('connected!')
        elif routeKey == '$disconnect':
            print('disconnected!')
        else:
            body = event.get("body", "")
            #print("data[0:8]: ", body[0:8])
            if body[0:8] == "__ping__":
                # print("keep alive!")                
                sendMessage(connectionId, "__pong__")
            else:
                print('connectionId: ', connectionId)
                print('routeKey: ', routeKey)
        
                jsonBody = json.loads(body)
                print('request body: ', json.dumps(jsonBody))

                requestId  = jsonBody['request_id']
                try:
                    msg, reference = getResponse(connectionId, jsonBody)

                    print('msg+reference: ', msg+reference)
                                        
                except Exception:
                    err_msg = traceback.format_exc()
                    print('err_msg: ', err_msg)

                    sendErrorMessage(connectionId, requestId, err_msg)    
                    raise Exception ("Not able to send a message")

    return {
        'statusCode': 200
    }
