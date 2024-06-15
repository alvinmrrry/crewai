import streamlit as st
import os
from groq import Groq
import random
from crewai import Agent, Task, Crew, Process

from langchain.chains import ConversationChain, LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import langchain_community.tools.tavily_search.tool


groq_api_key = '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'
groq_llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")
TAVILY_API_KEY = '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'
search_tool = TavilySearchResults()

# Define your agents with roles and goals
researcher = Agent(
  role='Senior Research Analyst',
  goal='Uncover cutting-edge developments in AI and data science',
  backstory="""You work at a leading tech think tank.
  Your expertise lies in identifying emerging trends.
  You have a knack for dissecting complex data and presenting actionable insights.""",
  verbose=True,
  allow_delegation=False,
  # You can pass an optional llm attribute specifying what model you wanna use.
  llm=groq_llm,
  tools=[search_tool]
)
writer = Agent(
  role='Tech Content Strategist',
  goal='Craft compelling content on tech advancements',
  backstory="""You are a renowned Content Strategist, known for your insightful and engaging articles.
  You transform complex concepts into compelling narratives.""",
  verbose=True,
  llm=groq_llm,
  allow_delegation=True
)

# Create tasks for your agents
task1 = Task(
  description="""Conduct a comprehensive analysis of the latest advancements in AI in 2024.
  Identify key trends, breakthrough technologies, and potential industry impacts.""",
  expected_output="Full analysis report in bullet points",
  agent=researcher
)

task2 = Task(
  description="""Using the insights provided, develop an engaging blog
  post that highlights the most significant AI advancements.
  Your post should be informative yet accessible, catering to a tech-savvy audience.
  Make it sound cool, avoid complex words so it doesn't sound like AI.""",
  expected_output="Full blog post of at least 4 paragraphs",
  agent=writer
)

# Instantiate your crew with a sequential process
crew = Crew(
  agents=[researcher, writer],
  tasks=[task1, task2],
  verbose=2, # You can set it to 1 or 2 to different logging levels
)

# Get your crew to work!
result = crew.kickoff()

st.write("Chatbot:", result)

# def main():
#     """
#     This function is the main entry point of the application. It sets up the Groq client, the Streamlit interface, and handles the chat interaction.
#     """
    
#     # Get Groq API key
#     groq_api_key = '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'

#     # Display the Groq logo
#     spacer, col = st.columns([5, 1])  
#     with col:  
#         st.image('groqcloud_darkmode.png')

#     # The title and greeting message of the Streamlit application
#     st.title("Welcome to this AI tool!")
#     st.write("Let's start our conversation!")

#     # Add customization options to the sidebar
#     st.sidebar.title('Customization')
#     system_prompt = st.sidebar.text_input("System prompt:")
#     model = st.sidebar.selectbox(
#         'Choose a model',
#         ['llama3-8b-8192','llama3-70b-8192', 'mixtral-8x7b-32768', 'gemma-7b-it']
#     )
#     conversational_memory_length = st.sidebar.slider('Conversational memory length:', 1, 10, value = 5)

#     memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)

#     user_question = st.text_area("Please ask a question:",height=200)

#     # session state variable
#     if 'chat_history' not in st.session_state:
#         st.session_state.chat_history=[]
#     else:
#         for message in st.session_state.chat_history:
#             memory.save_context(
#                 {'input':message['human']},
#                 {'output':message['AI']}
#                 )


#     # Initialize Groq Langchain chat object and conversation
#     groq_chat = ChatGroq(
#             groq_api_key=groq_api_key, 
#             model_name=model
#     )


#     # If the user has asked a question,
#     if user_question:

#         # Construct a chat prompt template using various components
#         prompt = ChatPromptTemplate.from_messages(
#             [
#                 SystemMessage(
#                     content=system_prompt
#                 ),  # This is the persistent system prompt that is always included at the start of the chat.

#                 MessagesPlaceholder(
#                     variable_name="chat_history"
#                 ),  # This placeholder will be replaced by the actual chat history during the conversation. It helps in maintaining context.

#                 HumanMessagePromptTemplate.from_template(
#                     "{human_input}"
#                 ),  # This template is where the user's current input will be injected into the prompt.
#             ]
#         )

#         # Create a conversation chain using the LangChain LLM (Language Learning Model)
#         conversation = LLMChain(
#             llm=groq_chat,  # The Groq LangChain chat object initialized earlier.
#             prompt=prompt,  # The constructed prompt template.
#             verbose=True,   # Enables verbose output, which can be useful for debugging.
#             memory=memory,  # The conversational memory object that stores and manages the conversation history.
#         )
        
#         # The chatbot's answer is generated by sending the full prompt to the Groq API.
#         response = conversation.predict(human_input=user_question)
#         message = {'human':user_question,'AI':response}
#         st.session_state.chat_history.append(message)
#         st.write("Chatbot:", response)

# if __name__ == "__main__":
#     main()