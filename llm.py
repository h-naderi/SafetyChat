import cv2
import openai
from langchain.vectorstores import Chroma
import os
import json
from flask import Flask, request, jsonify, render_template, Response, make_response
from flask_cors import CORS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory,ConversationBufferWindowMemory
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain,RetrievalQA,LLMChain
from langchain.chat_models import ChatOpenAI
##print prompt
#import langchain
#langchain.debug = True
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# add token
openai.api_key = "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"

# setting for the app
app = Flask(__name__)
CORS(app)
# setting for model
embeddings = OpenAIEmbeddings(openai_api_key="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")
db3 = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)


# PROMPT4 = PromptTemplate.from_template(
# """
# History:{chat_history}
# You are a safety trainer. Create an open-ended dangerous scenario whick interacts with me based on the following steps: 
# 1- My task is {input} and the related regulation to my task is mentioned in the following regulation section 
# regulation: {context}
# 2-Create a scenario following features: a) maximum 100 words and no questions b) different from the History c) in a way that create doubts to me that what I have to do 
# 3-Begin your answer with ""Imagine as a construction worker, you are tasked to ..."

# Return the answer as a JSON object:
# {{
#  "Scenario" : String
# }}
# """
# )


PROMPT4 = PromptTemplate.from_template(
"""
You are a safety trainer. Follow these steps:
1- If History is empty, create a scenario simulating a dangerous situation based on the following:
   1-a- Your task is {input}, and the related regulation is mentioned in the following section:
        Regulation: {context}
   1-b- Keep it within 100 words and avoid questions.
   1-c- Create ambiguity about what actions need to be taken.
   1-d- Begin your answer with "Imagine as a construction worker, you are tasked to..."

2- If History is not empty, continue the previous scenario based on the following:
   2-a- The user's opinion regarding the first scenario is .
   2-b- If the user's opinion is correct, introduce a different dangerous situation.
   2-c- If the user's opinion is incorrect, consider the user's fault and the required regulations.
   2-d- Begin your answer with "Let's continue the previous scenario..."

History:{chat_history}

Return the answer as a JSON object:
{{
  "Scenario": String
}}
"""
)


PROMPT5 = PromptTemplate.from_template(
"""
Design three challenging options (max 30 words) to assess my safety behavior in the given situation: 
situation: {input}

Return the answer as a JSON object:
{{
    "A": String
    "B": String
    "C": String
    "Right_Answer": Uppercase Character
}}

"""
)
PROMPT6 = PromptTemplate.from_template(
"""
Scenario:{scenario}
OptionA:{optiona}
OptionB:{optionb}
OptionC:{optionc}
My choice is {choice}
Context: {context}
Right Answer: {rightanswer}
1-if My choice is correct only encourage me for another scenario
1-if I'm wrong give me a feedback considering my choice with the aim of improving my behavior based on Right Answer
2-Answer is 150 words
3-Start with "Your choice was ..."

Return the answer as a JSON object:
{{
    "Feedback": String
}}
"""
)
PROMPT8 = PromptTemplate.from_template(
"""
History:{chat_history}
Scenario:{scenario}
Feedback:{feedback}

You can use imformation above to Answer Human Input
Human Input{input}

Return the answer as a JSON object:
{{
    "AI":String
}}
"""
)

memory = ConversationBufferWindowMemory(k=5,memory_key='chat_history',input_key='input')
memory2 = ConversationBufferWindowMemory(k=5,memory_key='chat_history',input_key='input')

new1 = LLMChain(
    llm=OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0.5, openai_api_key=openai.api_key),
    #llm=ChatOpenAI(model_name='gpt-4', openai_api_key= openai.api_key, temperature = 0.5),
    verbose=True,
    prompt = PROMPT4,
    memory=memory
)
new2 = LLMChain(
    llm=OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0.1, openai_api_key=openai.api_key),
    #llm=ChatOpenAI(model_name='gpt-4', openai_api_key= openai.api_key, temperature = 0.1),
    verbose=True,
    prompt = PROMPT5
)
new3 = LLMChain(
    llm=OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0.5, openai_api_key=openai.api_key),
    #llm=ChatOpenAI(model_name='gpt-4', openai_api_key= openai.api_key, temperature = 0.5),
    verbose=True,
    prompt = PROMPT6
)

new5 = LLMChain(
    llm=OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0.5, openai_api_key=openai.api_key),
    #llm=ChatOpenAI(model_name='gpt-4', openai_api_key= openai.api_key, temperature = 0.5),
    verbose=True,
    prompt = PROMPT8,
    memory = memory2
)
###app part
@app.route('/')
def hello():
    return 'Hello World'

retriever=db3.as_retriever(search_kwargs={"k": 2})
@app.route("/scenario", methods=["POST"])
def scenario():
    user_projects = request.json.get("projects")
    retrieved_docs = retriever.invoke("protection equipments for " + user_projects)
    for i in range(10):
        try:
            prompt1 = new1.predict(input=user_projects,context = retrieved_docs[0].page_content + "\n" + retrieved_docs[1].page_content)
            loader = json.loads(prompt1)
            result = jsonify({"scenario": loader["Scenario"]})
        except:
            continue
        break
    return result

@app.route("/options", methods=["POST"])
def options():
    prompt1 = request.json.get("prompt")
    for i in range(10):
        try:
            response_message = new2.predict(input=prompt1)
            loader = json.loads(response_message)
            result = jsonify({"A": loader["A"],"B": loader["B"],"C": loader["C"],"Right_Answer": loader["Right_Answer"]})
        except:
            continue
        break
    return result

@app.route("/combinetwo", methods=["POST"])
def combinetwo():
    user_projects = request.json.get("projects")
    retrieved_docs = retriever.invoke("protection equipments for " +user_projects)
    for i in range(10):
        try:
            prompt1 = new1.predict(input=user_projects,context = retrieved_docs[0].page_content + "\n" + retrieved_docs[1].page_content)
            print(prompt1)
            loader = json.loads(prompt1)
            scenario = loader["Scenario"]
        except:
            continue
        break
    for i in range(10):
        try:
            response_message = new2.predict(input=scenario)
            
            print(response_message)
            loader2 = json.loads(response_message)
            result2 = jsonify({"scenario": scenario,"A": loader2["A"],"B": loader2["B"],"C": loader2["C"],"Right_Answer": loader2["Right_Answer"]})
        except:
            continue
        break
    return result2
    

@app.route("/feedback", methods=["POST"])
def feedback():
    scenario = request.json.get("scenario")
    optiona = request.json.get("optiona")
    optionb = request.json.get("optionb")
    optionc = request.json.get("optionc")
    choice = request.json.get("choice")
    right_answer= request.json.get("rightanswer")
    if right_answer == "A": 
        retrieved_docs = retriever.invoke(optiona)
    elif right_answer == "B":
        retrieved_docs = retriever.invoke(optionb)
    else:
        retrieved_docs = retriever.invoke(optionc)
    
    for i in range(10):
        try:
            response = new3.predict(rightanswer= right_answer, scenario = scenario,optiona=optiona,optionb = optionb,optionc = optionc,choice=choice,context = retrieved_docs[0].page_content + "\n" + retrieved_docs[1].page_content)
            loader = json.loads(response)
            result = jsonify({"Feedback": loader["Feedback"]})
        except:
            continue
        break
    return result

@app.route("/chat", methods=["POST"])
def chat():
    scenario = request.json.get("scenario")
    feedback = request.json.get("feedback")
    input = request.json.get("input")
    for i in range(10):
        try:
            response = new5.predict(scenario=scenario,feedback=feedback,input=input)
            loader = json.loads(response)
            result = jsonify({"AI": loader["AI"]})
        except:
            continue
        break
    return result
    
if __name__ == "__main__":
    app.run(host = '0.0.0.0', port=5011)
    