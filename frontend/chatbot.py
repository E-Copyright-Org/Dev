from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)

import json
try:
    # Load configuration from config.json
    with open('config.json') as config_file:
        config = json.load(config_file)
        api_key = config.get('openai_api_key', None)
        
    if api_key is None:
        print("API key is missing or incorrect. Please provide a valid API key in the config.json file.")
        exit(1)

    # Initialize chatbot model
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=api_key)

    # Initialize conversation chain with memory
    buffer_memory = ConversationBufferWindowMemory(k=5, return_messages=True)
    system_msg_template = SystemMessagePromptTemplate.from_template(
        template="""Answer the question as truthfully as possible using the provided context, 
        and if the answer is not contained within the text below, say 'I don't know'"""
    )
    human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")
    prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])
    conversation = ConversationChain(memory=buffer_memory, prompt=prompt_template, llm=llm, verbose=True)

    # Chat loop
    while True:
        # Get user input
        user_input = input("You: ")

        # Process user input
        try:
            response = conversation.predict(input=user_input)
            print("Bot:", response)
        except Exception as e:
            print("An error occurred during conversation:", e)

except Exception as e:
    print("An error occurred during initialization:", e)




