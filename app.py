from crewai.project import CrewBase
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq

@CrewBase
class FinancialAnalystCrew(Crew):
    """FinancialAnalystCrew crew"""
    agents_config = 'config/agents.yml'
    task_config = 'config/tasks.yml'

    def __init__(self) -> None:
        self.groq_llm = ChatGroq(temperature=0.0, model_name="llama3-8b-8192")

    @agent
    def company_researcher(self) -> Agent:
        """Company Researcher agent"""
        return Agent(
            config = self.agents_config['company_researcher'],
            llm = self.groq_llm
        )
    
    @agent
    def company_analyst(self) -> Agent:
        """Company Researcher agent"""
        return Agent(
            config = self.agents_config['company_analyst'],
            llm = self.groq_llm
        )
    
    @task
    def research_company_task(self) -> Task:
        """Research Company task"""
        return Task(
            config = self.task_config['research_company_task'],
            agent = self.company_researcher()
        )
    
    @task
    def analyse_company_task(self) -> Task:
        """analyse Company task"""
        return Task(
            config = self.task_config['analyse_company_task'],
            agent = self.company_analyst()
        )
    
    @crew
    def crew(self) -> Crew:
        """Creates the FinancialAnalystCrew crew"""
        return Crew(
           agents = self.agents,
           tasks = self.tasks,
           process = Process.sequential,
           verbose = 2
        )
    
def main():
    """Main function"""
    inputs = {
        'company_name': 'Tesla',
    }
    FinancialAnalystCrew().crew().kickoff(inputs=inputs)

if __name__ == '__main__':
    main()