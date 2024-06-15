import streamlit as st
from crewai.project import CrewBase, agent, task, crew
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq

groq_llm = ChatGroq(temperature=0.0, model_name="llama3-8b-8192")

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
#   tools=[search_tool]
)
writer = Agent(
  role='Tech Content Strategist',
  goal='Craft compelling content on tech advancements',
  backstory="""You are a renowned Content Strategist, known for your insightful and engaging articles.
  You transform complex concepts into compelling narratives.""",
  verbose=True,
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

print("######################")
print(result)
# @CrewBase
# class FinancialAnalystCrew(Crew):
#     """FinancialAnalystCrew crew"""
#     agents_config = 'config/agents.yml'
#     task_config = 'config/tasks.yml'

#     def __init__(self) -> None:
#         self.groq_llm = ChatGroq(temperature=0.0, model_name="llama3-8b-8192")

#     @agent
#     def company_researcher(self) -> Agent:
#         """Company Researcher agent"""
#         return Agent(
#             config = self.agents_config['company_researcher'],
#             llm = self.groq_llm
#         )
    
#     @agent
#     def company_analyst(self) -> Agent:
#         """Company Researcher agent"""
#         return Agent(
#             config = self.agents_config['company_analyst'],
#             llm = self.groq_llm
#         )
    
#     @task
#     def research_company_task(self) -> Task:
#         """Research Company task"""
#         return Task(
#             config = self.task_config['research_company_task'],
#             agent = self.company_researcher()
#         )
    
#     @task
#     def analyse_company_task(self) -> Task:
#         """analyse Company task"""
#         return Task(
#             config = self.task_config['analyse_company_task'],
#             agent = self.company_analyst()
#         )
    
#     @crew
#     def crew(self) -> Crew:
#         """Creates the FinancialAnalystCrew crew"""
#         return Crew(
#            agents = self.agents,
#            tasks = self.tasks,
#            process = Process.sequential,
#            verbose = 2
#         )
    
# def main():
#     """Main function"""
#     inputs = {
#         'company_name': 'Tesla',
#     }
#     FinancialAnalystCrew().crew().kickoff(inputs=inputs)

# if __name__ == '__main__':
#     main()