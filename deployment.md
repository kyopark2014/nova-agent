# 인프라 설치하기

## Bedrock 사용 권한 설정하기

LLM으로 Amazon Nova Pro을 사용하기 위하여, Amazon Bedrock의 us-west-2, us-east-1, us-east-2 리전을 사용합니다. [Model access](https://us-west-2.console.aws.amazon.com/bedrock/home?region=us-west-2#/modelaccess)에 접속해서 [Edit]를 선택하여 "Titan Text Embeddings V2"와 "Amazon Nova Pro"을 Vector Embedding과 LLM을 위해 enable 합니다.


## 설치하기

### EC2를 사용하여 설치 환경 구성하기

여기서는 편의상 us-west-2 (Oregon) 리전을 사용합니다.

#### EC2 생성

[EC2 - Launch an Instance](https://us-west-2.console.aws.amazon.com/ec2/home?region=us-west-2#LaunchInstances:)에 접속하여 Name으로 "chatbot"이라고 입력합니다.

![noname](https://github.com/user-attachments/assets/acdac538-ea1e-4b32-a7f8-efc2b0e34664)

OS로 기본값인 "Amazon Linux"를 유지하고, Amazon Machine Image (AMI)도 기본값을 그대로 사용합니다.

Instance Type은 "m5.large"를 선택하고, Key pair는 "Proceeding without a key pair"를 선택합니다. 

[Configure storage]는 편의상 80G로 변경하고 [Launch instance]를 선택하여 EC2를 설치합니다. 

![noname](https://github.com/user-attachments/assets/84edf46d-0aa8-478c-8727-1301cf32f4db)

이후 아래와 같이 instance를 선택하여 EC2 instance 화면으로 이동하거나, console에서 [EC-Instances](https://us-west-2.console.aws.amazon.com/ec2/home?region=us-west-2#Instances:)로 접속합니다. 

![noname](https://github.com/user-attachments/assets/f5c82338-3e05-4c26-bdef-642c81f2c5d2)

아래와 같이 instance에서 [Connect]를 선택하여 [Session Manager]로 접속합니다. 

#### 관련 패키지 설치

편의상 C-Shell로 변경후 필요한 패키지로 git, node.js, npm, docker를 설치하고 환경을 설절정합니다. 

```text
csh
cd && sudo yum install git nodejs npm docker -y
sudo usermod -a -G docker $USER
newgrp docker
sudo service docker start
sudo npm install -g aws-cdk --prefix /usr/local
```

<!--

## CDK를 이용한 인프라 설치하기

여기서는 [AWS Cloud9](https://aws.amazon.com/ko/cloud9/)에서 [AWS CDK](https://aws.amazon.com/ko/cdk/)를 이용하여 인프라를 설치합니다. 

1) [Cloud9 Console](https://us-west-2.console.aws.amazon.com/cloud9control/home?region=us-west-2#/create)에 접속하여 [Create environment]-[Name]에서 “chatbot”으로 이름을 입력하고, EC2 instance는 “m5.large”를 선택합니다. 나머지는 기본값을 유지하고, 하단으로 스크롤하여 [Create]를 선택합니다.

![image](https://github.com/kyopark2014/stream-chatbot-for-amazon-bedrock/assets/52392004/c85c2ef5-4f96-4528-b5d4-ab9d3e52324e)

2) [Environment](https://us-west-2.console.aws.amazon.com/cloud9control/home?region=us-west-2#/)에서 “chatbot”를 [Open]한 후에 아래와 같이 터미널을 실행합니다.

![image](https://github.com/kyopark2014/stream-chatbot-for-amazon-bedrock/assets/52392004/fcf24f93-9ab3-4905-be8d-8146c7371951)

3) EBS 크기 변경

아래와 같이 스크립트를 다운로드 합니다. 

```text
curl https://raw.githubusercontent.com/kyopark2014/technical-summary/main/resize.sh -o resize.sh
```

이후 아래 명령어로 용량을 80G로 변경합니다.
```text
chmod a+rx resize.sh && ./resize.sh 80
```
-->


### 소스 다운로드 및 설치 

1) 소스를 다운로드합니다.

```java
git clone https://github.com/kyopark2014/nova-agent
```

2) cdk 폴더로 이동하여 필요한 라이브러리를 설치합니다.

```java
cd nova-agent/cdk-nova-agent/ && npm install
```

3) CDK 사용을 위해 Boostraping을 수행합니다.

아래 명령어로 Account ID를 확인합니다.

```java
aws sts get-caller-identity --query Account --output text
```

아래와 같이 bootstrap을 수행합니다. 여기서 "account-id"는 상기 명령어로 확인한 12자리의 Account ID입니다. bootstrap 1회만 수행하면 되므로, 기존에 cdk를 사용하고 있었다면 bootstrap은 건너뛰어도 됩니다.

```java
cdk bootstrap aws://[account-id]/us-west-2
```

4) 아래 명령어로 인프라를 설치합니다.

```java
cdk deploy --require-approval never --all
```

인프라가 설치가 되면 아래와 같은 Output을 확인할 수 있습니다. 

![noname](https://github.com/user-attachments/assets/7952f4f4-950d-4999-8411-1a629acffecd)


5) Output의 HtmlUpdateCommend을 아래와 같이 복사하여 실행합니다.

![noname](https://github.com/user-attachments/assets/48824379-b2ad-4377-916b-82988be90182)

    

6) Hybrid 검색을 위한 Nori Plug-in 설치

[OpenSearch Console](https://us-west-2.console.aws.amazon.com/aos/home?region=us-west-2#opensearch/domains)에서 "nova-agent"로 들어가서 [Packages] - [Associate package]을 선택한 후에, 아래와 같이 "analysis-nori"을 설치합니다. 

![image](https://github.com/user-attachments/assets/9297a93a-cf25-4fea-aae1-8b6b00e79949)

7) API에 대한 Credential을 획득하고 입력합니다.

- 일반 검색을 위하여 [Tavily Search](https://app.tavily.com/sign-in)에 접속하여 가입 후 API Key를 발급합니다. 이것은 tvly-로 시작합니다.

Tavily의 경우 1000건/월을 허용하므로 여러 건의 credential을 사용하면 편리합니다. 따라서, 아래와 같이 array형태로 입력합니다. 

```java
["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","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"]
```
  
- 날씨 검색을 위하여 [openweathermap](https://home.openweathermap.org/api_keys)에 접속하여 API Key를 발급합니다.
- [langsmith.md](https://github.com/kyopark2014/langgraph-agent/blob/main/langsmith.md)를 참조하여 [LangSmith](https://www.langchain.com/langsmith)에 가입후 API Key를 발급 받습니다.

[Secret manager](https://us-west-2.console.aws.amazon.com/secretsmanager/listsecrets?region=us-west-2)에 접속하여, [openweathermap-langgraph-agent](https://us-west-2.console.aws.amazon.com/secretsmanager/secret?name=openweathermap-langgraph-agent&region=us-west-2), [tavilyapikey-langgraph-agent](https://us-west-2.console.aws.amazon.com/secretsmanager/secret?name=tavilyapikey-langgraph-agent&region=us-west-2), [langsmithapikey-langgraph-agent](https://us-west-2.console.aws.amazon.com/secretsmanager/secret?name=langsmithapikey-langgraph-agent&region=us-west-2)에 접속하여, [Retrieve secret value]를 선택 후, api key를 입력합니다.

8) Output의 WebUrlforstreamchatbot의 URL로 접속합니다. 만약 Credential을 입력 전에 URL을 접속을 했다면, Lambda를 재배포하거나 일정 시간후에 Lamba가 내려갈때까지 기다렸다가 재접속하여야 하므로, Credential들을 입력 후에 URL로 접속하는것이 좋습니다. 

