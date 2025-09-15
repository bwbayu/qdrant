from agents.src.decision_crew.tools.custom_tool import VideoEmbedTool, CogneeTool, ChunksTool, SummariesTool, RAGCompletionTool, GraphCompletionTool
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from pydantic import BaseModel, Field
from typing import List, Dict
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators


class ResearchSource(BaseModel):
    title: str = Field(description="Title of the source")
    type: str = Field(
        description="Type of the source, e.g., 'video', 'cognee tool', etc.")
    link: str = Field(
        description="URL of the source just copy paste from context")
    transcription: str = Field(
        description="transcription or description of the source just copy paste from context")
    start_offset_sec: str = Field(description="Start offset in seconds")
    end_offset_sec: str = Field(description="End offset in seconds")
    embedding_option: str = Field(
        description="Embedding option, e.g., 'visual-text'")
    embedding_scope: str = Field(description="Embedding scope, e.g., 'clip'")


class ResearchPoint(BaseModel):
    topic: str = Field(description="The main topic or area being discussed")
    findings: str = Field(
        description="The key findings or insights about this topic")
    relevance: str = Field(
        description="Why this finding is relevant or important")
    sources: List[ResearchSource] = Field(
        description="Sources with title and URL for each finding, if video insert start_offset_sec and end_offset_sec",
        # default_factory=list
    )


class ResearchOutput(BaseModel):
    research_points: List[ResearchPoint] = Field(
        description="List of research findings")
    summary: str = Field(description="Brief summary of overall findings")


class ExecutiveReportSection(BaseModel):
    section_emoji: str = Field(description="Section emoji (e.g., 🔍, 📊, 🎯)")
    section_title: str = Field(description="Section title")
    section_content: str = Field(description="Main content of the section")
    # key_insights: List[str] = Field(
    #     description="Key insights from this section")
    # recommendations: List[str] = Field(
    #     default_factory=list,
    #     description="Optional recommendations based on findings"
    # )
    sources: List[Dict[str, str]] = Field(
        description="Sources with title and URL for this section",
    )


class SourceMetadata(BaseModel):
    link: str = Field(default="", description="URL of the source")
    transcription: str = Field(default="",
                               description="transcription or description of the source")
    start_offset_sec: str = Field(
        default="", description="Start offset in seconds")
    end_offset_sec: str = Field(
        default="", description="End offset in seconds")
    embedding_option: str = Field(
        default="",
        description="Embedding option, e.g., 'visual-text'")

    embedding_scope: str = Field(default="",
                                 description="Embedding scope, e.g., 'clip'")


class ExecutiveReport(BaseModel):
    report_title: str = Field(default="", description="Title of the report")
    generation_date: str = Field(
        default="", description="Report generation date")
    summary: str = Field(
        default="", description="A concise summary of all findings")
    # key_findings: List[Dict[str, str]] = Field(
    #     description="List of key findings with their sources",
    #     default_factory=list
    # )
    lists: List[SourceMetadata] = Field(
        default=[],
        description="Detailed lists of metadata sources(copy paste from context)",
    )
    # report_sections: List[ExecutiveReportSection] = Field(
    #     description="Detailed report sections"
    # )
    # sources: List[Dict[str, str]] = Field(
    #     description="All sources used in the report",
    #     default_factory=list
    # )


@CrewBase
class DecisionCrew():
    """DecisionCrew crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended

    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'],  # type: ignore[index]
            verbose=True,
            # Example of adding tools to an agent
            tools=[
                VideoEmbedTool(),
                ChunksTool(),
                SummariesTool(),
                RAGCompletionTool(),
                GraphCompletionTool(),
            ]
        )

    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(
            # type: ignore[index]
            config=self.agents_config['reporting_analyst'],
            verbose=True
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'],  # type: ignore[index]
            output_json=ResearchOutput
        )

    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config['reporting_task'],  # type: ignore[index]
            # output_file='report.md',
            output_json=ExecutiveReport
        )

    @crew
    def crew(self) -> Crew:
        """Creates the DecisionCrew crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
