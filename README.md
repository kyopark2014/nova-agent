# Amazon Nova와 Agentic Worfflow를 이용하여 복잡한 Application 구현하기

여기에서는 Amazon의 Nova와 Agentic Workflow를 이용하여 복잡한 Application을 구현하는것을 설명합니다.

## Basic Chat

[Nova Prompt](https://docs.aws.amazon.com/nova/latest/userguide/prompting-precision.html)를 참조하여, system과 human prompt를 정의하고 history까지 포함하여 invoke 한 후에 결과를 stream으로 client로 전송합니다. Nova Pro에 맞게 model_id와 parameter를 지정합니다. 

```python
chat = ChatBedrock(  
  model_id="us.amazon.nova-pro-v1:0",
  client=boto3_bedrock, 
  model_kwargs=parameters,
) 

system = (
   "당신은 사려깊은 인공지능 도우미입니다." 
   "당신은 상황에 맞는 구체적인 세부 정보를 충분히 제공합니다." 
   "너의 이름은 서연이고, 모르는 질문을 받으면 솔직히 모른다고 말합니다."
   "답변은 markdown 포맷을 사용하지 않고 text 형태로 제공합니다."
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

## Multi-modal

## Agent


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


![noname](https://github.com/user-attachments/assets/b692d664-2d93-402f-a0ca-a9533a1cd91f)



### Multi-modal 시험

아래의 Architecture를 다운로드하여 채팅창 하단의 파일 아이콘을 선택하여 업로드 합니다. 

<img width="640" alt="image" src="https://github.com/user-attachments/assets/5227d6ce-bef7-4b87-af19-870ed5488eb9">

이때의 결과는 아래와 같습니다.

![image](https://github.com/user-attachments/assets/7d4da2cc-9387-4976-b78a-ec0ec16c462d)

## 리소스 정리하기 

더이상 인프라를 사용하지 않는 경우에 아래처럼 모든 리소스를 삭제할 수 있습니다. 

1) [API Gateway Console](https://us-west-2.console.aws.amazon.com/apigateway/main/apis?region=us-west-2)로 접속하여 "rest-api-for-nova-agent", "ws-api-for-nova-agent"을 삭제합니다.

2) [Cloud9 Console](https://us-west-2.console.aws.amazon.com/cloud9control/home?region=us-west-2#/)에 접속하여 아래의 명령어로 전체 삭제를 합니다.

```text
cd ~/environment/nova-agent/cdk-nova-agent/ && cdk destroy --all
```
