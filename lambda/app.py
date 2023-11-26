import json
from typing import Tuple
from uuid import uuid4
import boto3
import pandas as pd
import io

from langchain import ConversationChain
from langchain.memory import ConversationBufferMemory, DynamoDBChatMessageHistory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chat_models import ChatOpenAI
from langchain.schema import messages_to_dict
from langchain.embeddings.openai import OpenAIEmbeddings

def obtain_csv_file_from_s3():
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket='hudi-poc11', Key='sample.xlsx')
    data = obj['Body'].read()
    df = pd.read_excel(io.BytesIO(data), sheet_name='Jakarta')
    df = df.to_csv()
    return df



def lambda_handler(event, context):
    api_key = event['api_key']
    session_id = event['session_id']
    prompt = event['prompt']
    df = obtain_csv_file_from_s3()
    print(df)
    if session_id == "":
        session_id = str(uuid4())
        print(session_id)
        chat_memory = DynamoDBChatMessageHistory(
            table_name="conversation-history-store",
            session_id=session_id
        )
        messages = chat_memory.messages
        memory = ConversationBufferMemory(chat_memory=chat_memory, return_messages=True)
        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "The following is a friendly conversation between a human and a content creator AI. The AI helps in "
                "creating content. The AI strictly gives answers only about events happening "
                "from this CSV data:{}".format(
                    df)),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])

        llm = ChatOpenAI(temperature=0, openai_api_key=api_key, model="gpt-4")
        conversation = ConversationChain(
            llm=llm,
            prompt=prompt_template,
            # verbose=True,
            memory=memory
        )

        response = conversation.predict(input=prompt)

        print(f"The response is {response}")
        print(f"The sesh_id is {session_id}")

    elif session_id:
        chat_memory = DynamoDBChatMessageHistory(
            table_name="conversation-history-store",
            session_id=session_id
        )
        messages = chat_memory.messages
        if messages:
            chat_memory = DynamoDBChatMessageHistory(
                table_name="conversation-history-store",
                session_id=session_id
            )


            df = obtain_csv_file_from_s3()

            memory = ConversationBufferMemory(chat_memory=chat_memory, return_messages=True)

            prompt_template = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(
                    "The following is a friendly conversation between a human and a content creator AI. The AI helps in"
                    " creating content. The AI strictly gives answers only about events happening"
                    " from this CSV data:{}".format(
                        df)),
                MessagesPlaceholder(variable_name="history"),
                HumanMessagePromptTemplate.from_template("{input}")
            ])

            llm = ChatOpenAI(temperature=0, openai_api_key=api_key, model_name="gpt-3.5-turbo")  # more temp = vivid answers
            conversation = ConversationChain(
                llm=llm,
                prompt=prompt_template,
                # verbose=True,
                memory=memory
            )

            response = conversation.predict(input=prompt)

            print(f"The response is {response}")
            print(f"The sesh_id is {session_id}")

        # return response, session_id


    return {
        "statusCode": 200,
        "body": json.dumps({
            "message": "hello world",
        }),
    }

# DDB Layout --> SessionId (String), History (String)