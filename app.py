import streamlit as st
from crewai import Agent, Task, Crew
from langchain.chains import ConversationChain, LLMChain
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.agents import tool

from crewai_tools import(
    DirectoryReadTool,
    FileReadTool,
    SerperDevTool,
    WebsiteSearchTool
)

SERPER_API_KEY = '2203d27aa32a1d92275134fb632bf009714b2476'
groq_api_key = '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'
groq_llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192", temperature=0)

# Instantiate tools
docs_tool = DirectoryReadTool(directory='src/financial_analyst_crew')
file_tool = FileReadTool()
search_tool = SerperDevTool()
web_rag_tool = WebsiteSearchTool()

@tool('DuckDuckGoSearch')
def search(query):
    """Search the web for informations """
    return DuckDuckGoSearchResults().run(query)

# Create agents
researcher = Agent(
    role='Market Research Analyst',
    goal='Provide up-to-date market analysis of the AI industry',
    backstory='An expert analyst with a keen eye for market trends.',
    llm = groq_llm,
    tools=[search_tool, web_rag_tool],
    verbose=True
)

writer = Agent(
    role='Content Writer',
    goal='Craft engaging blog posts about the AI industry',
    backstory='A skilled writer with a passion for technology.',
    llm = groq_llm,
    tools=[docs_tool, file_tool],
    verbose=True
)

# Define tasks
research = Task(
    description='Research the latest trends in the AI industry and provide a summary.',
    expected_output='A summary of the top 3 trending developments in the AI industry with a unique perspective on their significance.',
    agent=researcher
)

write = Task(
    description='Write an engaging blog post about the AI industry, based on the research analyst’s summary. Draw inspiration from the latest blog posts in the directory.',
    expected_output='A 4-paragraph blog post formatted in markdown with engaging, informative, and accessible content, avoiding complex jargon.',
    agent=writer,
    output_file='blog-posts/new_post.md'  # The final blog post will be saved here
)

# Assemble a crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[research, write],
    verbose=2
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