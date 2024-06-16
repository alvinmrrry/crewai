import streamlit as st
from crewai import Agent, Task, Crew
from langchain.chains import ConversationChain, LLMChain
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.agents import tool

groq_api_key = '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'
groq_llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")

@tool('DuckDuckGoSearch')
def search(**kwargs):
    """Search the web for informations """
    return DuckDuckGoSearchResults().run(**kwargs)

# Define your agents with roles and goals
researcher = Agent(
  role='Financial Researcher',
  goal='Gather all of the necessary information, using search tools, about a company for the financial analyst to prepare a report.',
  backstory="""An expert financial researcher, who spends all day and night thinking about finanacial performance of different companies.""",
  verbose=True,
  allow_delegation=False,
  # You can pass an optional llm attribute specifying what model you wanna use.
  llm=groq_llm,
  tools=[search]
)
writer = Agent(
  role='Financial Analyst',
  goal='Take provided company financial information and create a thorough financial report about a given company.',
  backstory=""" An expert financial analyst who prides themselves on creating clear and easily readable financial reports of different companies. """,
  verbose=True,
  llm=groq_llm,
  allow_delegation=True,
  tools=[search]
)

# Create tasks for your agents
task1 = Task(
  description="""Use a search tool to look up this company's stock information:Tesla.
    The goal is to prepare enough information to make an informed analysis of the company's stock performance.""",
  expected_output="All of the relevant financial information about the company's stock performance. ",
  agent=researcher
)

task2 = Task(
  description="""Take Tesla financial information and analyze it to make an informed analysis of the company's stock performance.
    The goal is to prepare a report that includes a summary of the company's financial performance and a recommendation for whether to buy, hold, or sell the company's stock.""",
  expected_output="A report that includes a summary of the company's financial performance and a recommendation for whether to buy, hold, or sell the company's stock. ",
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