# Nova Pro 활용하기

<p align="left">
    <a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fkyopark2014%2Fnova-agent&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com"/></a>
    <img alt="License" src="https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square">
</p>

여기에서는 [Nova Pro 모델](https://docs.aws.amazon.com/nova/latest/userguide/what-is-nova.html)을 활용하여 chat과 Hybrid 검색 지원하는 RAG를 구현하고, [CRAG](https://github.com/kyopark2014/langgraph-agent/blob/main/corrective-rag-agent.md), [Self RAG](https://github.com/kyopark2014/langgraph-agent/blob/main/self-rag.md), [Self Corrective RAG](https://github.com/kyopark2014/langgraph-agent/blob/main/self-corrective-rag.md)를 이용해 RAG의 성능을 향상시키는 방법에 대해 설명합니다. 또한 복잡한 application을 구현할 수 있는 agentic workflow의 4가지 대표적인 디자인 패턴인 [tool use](https://github.com/kyopark2014/langgraph-agent?tab=readme-ov-file#tool-use), [reflection](https://github.com/kyopark2014/langgraph-agent?tab=readme-ov-file#reflection), [planning](https://github.com/kyopark2014/langgraph-agent?tab=readme-ov-file#plan-and-execute), [multi-agent collaboration](https://github.com/kyopark2014/langgraph-agent?tab=readme-ov-file#multi-agent-collaboration)을 Nova Pro를 이용해 구현합니다. Workflow는 적절한 task들과 반복을 통해 LLM의 결과를 향상시키지만 하나의 job을 수행하기 위해 여러번 LLM을 호출하여야 하므로 지연시간이 증가합니다. Nova Pro는 다른 동급의 LLM 대비 2배 빠르고 가격은 1/3이므로 workflow 구현에 적절한 모델입니다.

## Architecture 개요

전체적인 Architecture는 아래와 같습니다. 속도 향상을 위해 Multi-region을 활용하고 OpenSearch를 이용해 RAG를 구성하고 인터넷 검색은 [Tavily](https://tavily.com/)를 이용합니다. 여기에서는 변화하는 트래픽과 유지보수 비용면에서 장점이 있는 서버리스를 활용합니다. 결과를 스트림으로 제공하기 위하여 WebSocket 지원하는 [API Gateway](https://docs.aws.amazon.com/ko_kr/apigateway/latest/developerguide/welcome.html)와 [AWS Lambda](https://docs.aws.amazon.com/lambda/latest/dg/welcome.html)를 이용해 API를 구성합니다. 여기에서는 [LangChain](https://python.langchain.com/docs/introduction/)을 이용해 Chat, Hybrid RAG, Translation을 구현하고, [LangGraph](https://langchain-ai.github.io/langgraph/)를 이용하여 CRAG, Self RAG, Self Corrective RAG와 agentic workflow의 4가지 패턴인 tool use, planning, reflection, multi-agent collaboration을 구현합니다.

<img src="https://github.com/user-attachments/assets/27afdb89-fc2e-4dc1-a2e1-f1423cbf685c" width="800">

## Chat의 구현 

[Nova Prompt](https://docs.aws.amazon.com/nova/latest/userguide/prompting-precision.html)를 참조하여, system과 human prompt를 정의하고 history까지 포함하여 invoke 한 후에 결과를 stream으로 client로 전송합니다. Nova Pro에 맞게 model_id와 parameter를 지정합니다. 

```python
chat = ChatBedrock(  
  model_id="us.amazon.nova-pro-v1:0",
  client=boto3_bedrock, 
  model_kwargs=parameters,
) 

system = (
  "당신의 이름은 서연이고, 질문에 대해 친절하게 답변하는 사려깊은 인공지능 도우미입니다."
  "상황에 맞는 구체적인 세부 정보를 충분히 제공합니다." 
  "모르는 질문을 받으면 솔직히 모른다고 말합니다."
)    
human = "{input}"

prompt = ChatPromptTemplate.from_messages([("system", system), MessagesPlaceholder(variable_name="history"), ("human", human)])
history = memory_chain.load_memory_variables({})["chat_history"]
          
chain = prompt | chat    

isTyping(connectionId, requestId, "")  
stream = chain.invoke(
   {
       "history": history,
       "input": query,
   }
)

msg = ""
for event in stream:
    msg = msg + event

    result = {
        'request_id': requestId,
        'msg': msg,
        'status': 'proceeding'
    }
    sendMessage(connectionId, result)
```

## RAG

RAG를 통해 얻어진 문서는 아래와 같이 LangChain을 이용해 context로 활용되고 적절한 응답을 얻을 수 있습니다. 

```python
system = (
  "당신의 이름은 서연이고, 질문에 대해 친절하게 답변하는 사려깊은 인공지능 도우미입니다."
  "다음의 Reference texts을 이용하여 user의 질문에 답변합니다."
  "모르는 질문을 받으면 솔직히 모른다고 말합니다."
  "답변의 이유를 풀어서 명확하게 설명합니다."
)
human = (
    "Question: {input}"

    "Reference texts: "
    "{context}"
)    
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
chain = prompt | chat
stream = chain.invoke(
    {
        "context": context,
        "input": revised_question,
    }
)
msg = readStreamMsg(connectionId, requestId, stream.content)    
```

### Corrective RAG

[Corrective RAG(CRAG)](https://github.com/kyopark2014/langgraph-agent/blob/main/corrective-rag-agent.md)는 retrival/grading 후에 질문을 rewrite한 후 인터넷 검색에서 얻어진 결과로 RAG의 성능을 강화하는 방법입니다. 

![image](https://github.com/user-attachments/assets/27228159-b307-4588-8a8a-61d8deaa90e3)

CRAG의 workflow는 아래와 같습니다. 

```python
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
```

### Self RAG

[Self RAG](https://github.com/kyopark2014/langgraph-agent/blob/main/self-rag.md)는 retrieve/grading 후에 generation을 수행하는데, grading의 결과에 따라 필요시 rewtire후 retrieve를 수행하며, 생성된 결과가 hallucination인지, 답변이 적절한지를 판단하여 필요시 rewtire / retrieve를 반복합니다. 

![image](https://github.com/user-attachments/assets/b1f2db6c-f23f-4382-86f6-0fa7d3fe0595)

Self RAG의 workflow는 아래와 같습니다.

```python
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
```

### Self Corrective RAG

Self Corrective RAG는 Self RAG처럼 retrieve / generate 후에 hallucination인지 답변이 적절한지 확인후 필요시 질문을 rewrite하거나 인터넷 검색을 통해 RAG의 성능을 향상시키는 방법입니다. 

![image](https://github.com/user-attachments/assets/9a18f7f9-0249-42f7-983e-c5a7f9d18682)

Self Corrective RAG의 workflow는 아래와 같습니다. 

```python
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
```

## Agent

여기에서는 LangGraph를 이용하여 tool use, reflection, planning, multi-agnet collaboration과 같은 패턴을 구현하고 동작을 비교합니다. 

### Tool Use

Tool use의 workflow는 아래와 같습니다. 이것은 다양한 tool을 이용해 자연스러운 대화를 수행할 수 있어서 가장 일반적으로 활용되는 agent 구현 패턴입니다. 

![image](https://github.com/user-attachments/assets/5be0a600-1e21-43f3-9af4-c3f65dccb4cc)

Toll use의 workflow는 아래와 같이 구현할 수 있습니다. 여기에서 agent와 action은 execution_agent_node와 tool_node로 구성됩니다. agent의 reasoning 결과가 tool_calls 인 경우에는 해당되는 tool을 실행하고, content인 경우에는 종료하고 결과를 final_answer 노드로 보내서 답변을 생성합니다. 

```python
workflow = StateGraph(State)
workflow.add_node("agent", execution_agent_node)
workflow.add_node("action", tool_node)
workflow.add_node("final_answer", final_answer)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": "final_answer",
    },
)
workflow.add_edge("action", "agent")
workflow.add_edge("final_answer", END)
```

API 처리를 이해하기 위해 "서울과 부산의 현재 날씨를 비교해주세요."라고 입력하면 Nova Pro의 경우에 reasoning 결과로 아래의 2개 API를 호출하게 됩니다. Claude Sonnet은 reasoning로 매번 1개의 action을 줌으로써 reasoning - action 동작을 2회 수행하지만, Nova Pro는 가능하다면 한번에 2개 API를 호출할 수 있도록 아래와 같은 응답을 제공합니다.

```java
[
   {
      "type":"tool_use",
      "name":"get_weather_info",
      "input":{
         "city":"서울"
      },
      "id":"tooluse_Kg9LCHHrR1mWz7VKHp0pmg"
   },
   {
      "type":"tool_use",
      "name":"get_weather_info",
      "input":{
         "city":"부산"
      },
      "id":"tooluse_B1sjTzWOQ8Kgcjuxhw28Sw"
   }
]
```

Claude Sonnet의 Reasoning 결과는 아래와 같습니다.

```java
[
   {
      "name":"get_weather_info",
      "args":{
         "city":"서울"
      },
      "id":"toolu_bdrk_01NWq2euSMtLwrE9HPDq7WzW",
      "type":"tool_call"
   }
]
```



### Reflection

Reflection 패턴은 초안을 생성한 후에 개선할 사항을 추출하고 추가 검색을 통해 얻어진 정보를 이용해 향상된 답변을 생성합니다. 따라서 아래와 같이 generate, reflect, revise_answer 노드들을 구성해 workflow를 생성합니다. 

![image](https://github.com/user-attachments/assets/a2b15e31-c727-41c9-9857-9e6082d05811)

Reflection의 workflow는 아래와 같습니다. generate 노드에서 생성된 초안(draft)는 reflect에서 수정할 부분과 추가 검색어를 추출합니다. 이후 revise_answer 노드에서는 draft를 개선합니다. 

```python
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
```

### Planning

Planing 패턴을 이용하면, CoT(Chain of Thought)형태로 반복적으로 결과를 개선함으로써 향상된 결과를 얻을 수 있습니다. 

![image](https://github.com/user-attachments/assets/4c0086da-865c-44c3-84fa-64246a10f624)

Planning agent의 workflow는 아래와 같습니다. plan 노드에서 최초 수행할 계획을 세우고, execution 노드에서 첫번째 plan을 실행한 후에 replan에서 업데이트된 plan을 생성합니다. plan/replan 과정을 통해 지속적으로 개선된 flow를 수행하고 축적된 결과들을 이용해 최종 답변을 구합니다. 

```python
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
```

### Multi-agent Collaboration

Multi-agent collaboration의 예로서 긴글을 쓰는 애플리케이션을 만들고자 합니다. 이때의 workflow는 아래와 같습니다.

![image](https://github.com/user-attachments/assets/ac783a78-b0af-4b69-9219-60fdab05202e)

여기에서는 planning agent와 reflection agent를 이용해 충분한 contenxt를 가지는 글을 작성합니다. 먼저 planning agnet의 workflow는 아래와 같습니다. plan 노드에서는 글 작성에 필요한 계획을 세우고, execution 노드에서는 초안을 생성합니다. 생성된 초안은 revise_answer 노드에서 reflection을 통해 업데이트 됩니다. 

```python
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
```

Refelection agent의 workflow는 아래와 같습니다. reflection 노드에서는 초안을 개선할 포인트와 추가 검색어를 추출하고 revise_draft 노드에서는 결과를 업데이트 합니다. 

```python
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
```


### Claude Sonnet과 Nova Pro의 Agent 동작 비교

"서울과 부산의 현재 날씨를 비교해주세요."와 같이 2번의 weather a;k 호출이 필요한 경우에, Claude Sonnet의 실행 결과는 아래와 같습니다.

![image](https://github.com/user-attachments/assets/f7757304-fba2-4374-9996-3b9bfd6ffe26)

이때의 Claude Sonnet으로 만든 Agent의 동작은 아래와 같습니다. 전체 동작에 21.5초가 소요되었고, reasoning - action - reasoning - action의 형태로 동작하였습니다. 

![noname](https://github.com/user-attachments/assets/d61d1de6-1b1d-4a79-8944-a29d803c7c16)


동일한 질문에 대한 Nova Pro의 결과는 아래와 같습니다. 여기서 온도 정보가 부정확한것은 [weather api](https://openweathermap.org/)가 한국 정보를 충분히 제공하지 못하기 때문입니다. Claude Sonnet과 Nova Pro의 결과는 유사합니다.

![image](https://github.com/user-attachments/assets/3d87d754-3ebc-44a5-8a5e-d74c0dd1c9fb)

이때의 동작은 아래와 같습니다. 전체 수행시간은 8.7초가 소요되었고, reasoning - action - action 형태로 weather api를 2회 연속 호출하였습니다. Nova Pro의 추론 속도는 Claude Sonnet 대비 2배 빠를 뿐 아니라, agent에서 API를 호출할 때에 연속적으로 action을 수행할 수 있어서, Claude Sonnet으로 만든 agent 대비 구조적으로 더 속도를 개선할 수 있습니다.

![noname](https://github.com/user-attachments/assets/12279b91-dd9c-447b-abe1-c0adcb6a960b)


## 직접 실습 해보기

### 사전 준비 사항

이 솔루션을 사용하기 위해서는 사전에 아래와 같은 준비가 되어야 합니다.

- [AWS Account 생성](https://repost.aws/ko/knowledge-center/create-and-activate-aws-account)에 따라 계정을 준비합니다.

### CDK를 이용한 인프라 설치

본 실습에서는 us-west-2 리전을 사용합니다. [인프라 설치](./deployment.md)에 따라 CDK로 인프라 설치를 진행합니다. 

## 실행결과

### RAG 구현

Amazon S3에 파일 업로드하면 자동으로 파싱하여 OpenSearch로 구성된 RAG에 chunk된 문서가 저장됩니다. 채팅창에 "LLM Ops에 대해 설명해주세요."라고 입력후에 결과를 확인합니다. 결과를 보면 다른 LLM들에 비하여 약 2배정도 결과를 제공합니다.

![noname](https://github.com/user-attachments/assets/f8fa8ccd-5a50-454b-beb0-197cf6af1b48)


### Agent

#### Reflection

Reasoning을 통해 tool로 search가 선택되였고, '지방 조직 exosome 면역체계 역할'와 '지방 조직 exosome 당뇨 예방'을 검색하여 초안(draft)을 작성합니다. 초안은는 아래와 같고, 전체 길이는 443자입니다.

```text
지방 조직이 분비하는 exosome들은 면역 체계에 다양한 역할을 할 수 있습니다. exosome은 세포 간 통신을 위한 중요한 매개체로, 다양한 생물학적 물질을 운반하여 면역 반응을 조절할 수 있습니다. 예를 들어, exosome은 면역 세포의 활성화, 염증 반응의 조절, 그리고 면역 기억의 형성에 관여할 수 있습니다. 
exosome을 통해 전달되는 신호 분자들은 면역 세포의 기능을 조절하여 감염이나 염증에 대한 반응을 조절할 수 있습니다. 또한, exosome은 면역 세포 간의 상호작용을 촉진하여 면역 반응을 강화하거나 억제할 수 있습니다.
좋은 exosome을 분비하여 당뇨나 질병을 예방하는 방법에 대해서는 현재까지 명확한 연구 결과가 없습니다. 그러나 일반적으로 건강한 생활 습관을 유지하고, 균형 잡힌 식사를 하며, 규칙적인 운동을 하는 것이 면역 체계를 강화하고 질병 예방에 도움이 될 수 있습니다. 또한, 스트레스 관리와 충분한 수면도 면역 체계의 건강에 중요한 역할을 합니다.
참고로, exosome의 역할과 면역 체계에 대한 연구는 지속적으로 진행되고 있으며, 향후 더 많은 연구 결과가 나올 것으로 기대됩니다.
```

Reflection에서 도출된 개선 사항입니다. 

```text
reflection:
missing='더 구체적인 exosome의 면역 체계에서의 역할과 당뇨나 질병 예방에 대한 연구 결과,
advisable='exosome의 면역 체계에서의 역할과 당뇨나 질병 예방에 대한 최신 연구 결과를 찾아보는 것이 좋을 것 같습니다.' 
superfluous='건강한 생활 습관, 균형 잡힌 식사, 규칙적인 운동, 스트레스 관리, 충분한 수면에 대한 설명은 exosome의 면역 체계에서의 역할과 당뇨나 질병 예방에 대한 연구 결과와 직접적인 관련이 없습니다.'

search_queries: 'exosome의 면역 체계에서의 역할에 대한 최신 연구 결과', 'exosome을 통한 당뇨나 질병 예방에 대한 연구 결과', 'exosome과 면역 체계의 상호작용에 대한 최신 연구 동향'
```

개선사항을 반영한 답변입니다. 전체 길이는 572자 입니다. 초안를 reflection를 이용해 개선하였고 길이도 30%정도 증가하였습니다.

![noname](https://github.com/user-attachments/assets/c040ca53-ee72-4358-a4a2-2d4ef47199d3)

전체 동작을 LangSmith를 이용해 확인하면 아래와 같습니다. 전체적으로 47초가 소요되었습니다. 먼저 reasoning을 통해 질문에서 2개의 검색어를 추출하여 tavily로 검색을 한 후에 초안을 생성하였습니다. 이후 개선할 사항을 추출하고 이를 반영하기 위하여 3회 추가 검색을 수행하여 최종 답변을 생성하였습니다.

![noname](https://github.com/user-attachments/assets/5c81d318-34eb-4948-ab22-42c8502b750f)


이번에는 "Amazon에서 SA로 일하는것"라고 입력하고 결과를 확인합니다.

![noname](https://github.com/user-attachments/assets/f3bc351e-acb6-462e-a86b-9a33449b6013)


이때의 동작을 LangSmith로 확인합니다. 여기에서는 3번의 검색을 통해, 초안(draft)를 생성한 후에 3회 추가 검색을 통해 초안의 답변을 향상시켰습니다.

![noname](https://github.com/user-attachments/assets/426e7a43-2f0a-4eb1-a5f7-14148dab74f4)



#### Planning

"LLM Ops에 대해 설명해줘"라고 입력시 처음 생성된 plan은 아래와 같습니다.

```text
planning_steps:

1. LLM Ops의 개념과 중요성을 파악합니다.

2. LLM Ops의 주요 구성 요소와 기능을 설명합니다.

3. LLM Ops의 구현 방법과 도구에 대해 설명합니다.

4. LLM Ops의 장점과 잠재적인 문제점을 분석합니다.
```

plan을 먼저 만든 후에, 첫번째 execute를 하고 이후로 replan을 반복하면서 원하는 답변을 찾습니다.

이때의 결과는 아래와 같습니다.

![noname](https://github.com/user-attachments/assets/91457669-bf2a-422d-887a-ddc3ca763fdd)

이를 실행을 보면 아래와 같이 plan / execute / replan의 과정을 통해 답변을 얻었음을 알 수 있습니다.

![noname](https://github.com/user-attachments/assets/8f1e5860-ebf1-402f-880d-28da222197e3)


### Multi-modal 시험

아래의 Architecture를 다운로드하여 채팅창 하단의 파일 아이콘을 선택하여 업로드 합니다. 

<img width="640" alt="image" src="https://github.com/user-attachments/assets/5227d6ce-bef7-4b87-af19-870ed5488eb9">

이때의 결과는 아래와 같습니다.

![image](https://github.com/user-attachments/assets/7d4da2cc-9387-4976-b78a-ec0ec16c462d)


### Multi-agent Collaboration

채팅창에서 "지방 조직이 분비하는 exosome들이 어떻게 면역체계에 역할을 하고 어떻게 하면 좋은 exosome들을 분비시켜 당뇨나 병을 예방할수 있는지 알려주세요."라고 입력후 결과를 확인합니다. 아래는 [Multi-agent Collaboration의 결과](./contents/지방_exosome의_면역_역할과_예방_방법.md)의 일부분입니다. 7067자(글자만 5446)의 답변이 생성되었습니다.
![noname](https://github.com/user-attachments/assets/34e6e9b5-bc3e-4607-85a6-b7bb7931e1b3)

이때 동작시간을 확인하면 아래와 같습니다.

![image](https://github.com/user-attachments/assets/5daa6eb3-7b0c-41d3-9f08-1ee6c974252e)

이때, 사용된 입력과 출력 token의 숫자는 각각 19,328과 7,787입니다. 

Multi-agent collaboration을 이용해 작성된 보고서의 예는 아래와 같습니다. 

- [지방 조직이 분비하는 exosome들이 어떻게 면역체계에 역할을 하고 어떻게 하면 좋은 exosome들을 분비시켜 당뇨나 병을 예방할수 있는지 알려주세요.](./contents/지방_exosome의_면역_역할과_예방_방법.md)

- [여수 여행](./contents/여수_여행_정보_및_추천.md)

- [제주 여행](./contents/Jeju_travel_experience.md)

## 리소스 정리하기 

더이상 인프라를 사용하지 않는 경우에 아래처럼 모든 리소스를 삭제할 수 있습니다. 

1) [API Gateway Console](https://us-west-2.console.aws.amazon.com/apigateway/main/apis?region=us-west-2)로 접속하여 "rest-api-for-nova-agent", "ws-api-for-nova-agent"을 삭제합니다.

2) [Cloud9 Console](https://us-west-2.console.aws.amazon.com/cloud9control/home?region=us-west-2#/)에 접속하여 아래의 명령어로 전체 삭제를 합니다.

```text
cd ~/nova-agent/cdk-nova-agent && cdk destroy --all
```
