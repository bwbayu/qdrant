from agents.src.decision_crew.tools.custom_tool import (
    VideoEmbedTool, CogneeTool, ChunksTool, SummariesTool,
    RAGCompletionTool, GraphCompletionTool
)
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List, Dict
from pydantic import BaseModel, Field

# =======================================================
# Data Models for structured input/output
# =======================================================


class ResearchSource(BaseModel):
    """Represents a single research source (video, tool output, etc.)."""
    title: str = Field(description="Title of the source")
    type: str = Field(
        description="Type of the source, e.g., 'video', 'cognee tool'")
    link: str = Field(description="URL of the source, copy-paste from context")
    transcription: str = Field(
        description="Full transcription/description, copy-paste from context")
    start_offset_sec: str = Field(description="Start offset in seconds")
    end_offset_sec: str = Field(description="End offset in seconds")
    embedding_option: str = Field(
        description="Embedding option, e.g., 'visual-text'")
    embedding_scope: str = Field(description="Embedding scope, e.g., 'clip'")


class ResearchPoint(BaseModel):
    """Represents a single research finding with related metadata."""
    topic: str = Field(description="The main topic or area being discussed")
    findings: str = Field(
        description="The key findings or insights about this topic")
    relevance: str = Field(
        description="Why this finding is relevant or important")
    sources: List[ResearchSource] = Field(
        description="Sources with title/URL for this finding. "
                    "If video, include start_offset_sec and end_offset_sec."
    )


class ResearchOutput(BaseModel):
    """Structured output for the research task."""
    research_points: List[ResearchPoint] = Field(
        description="List of research findings")
    summary: str = Field(description="Brief summary of overall findings")


class SourceMetadata(BaseModel):
    """Metadata wrapper for sources used in the final executive report."""
    link: str = Field(default="", description="URL of the source")
    transcription: str = Field(
        default="", description="Full transcription of the source")
    start_offset_sec: str = Field(
        default="", description="Start offset in seconds")
    end_offset_sec: str = Field(
        default="", description="End offset in seconds")
    embedding_option: str = Field(
        default="", description="Embedding option, e.g., 'visual-text'")
    embedding_scope: str = Field(
        default="", description="Embedding scope, e.g., 'clip'")


class ExecutiveReport(BaseModel):
    """Structured output for the reporting task (final executive report)."""
    report_title: str = Field(default="", description="Title of the report")
    generation_date: str = Field(
        default="", description="Report generation date")
    summary: str = Field(
        default="", description="Concise summary of all findings")
    lists: List[SourceMetadata] = Field(
        default=[],
        description="Detailed list of metadata sources (copy-paste from context)"
    )
    # Optional fields like detailed sections and sources could be added here


# =======================================================
# Crew Definition
# =======================================================

@CrewBase
class DecisionCrew():
    """Defines the DecisionCrew, consisting of agents and tasks."""

    # Agents and tasks will be created dynamically by CrewAI decorators
    agents: List[BaseAgent]
    tasks: List[Task]

    # -------------------
    # Agents
    # -------------------

    @agent
    def researcher(self) -> Agent:
        """Research agent that uses multiple tools to gather findings."""
        return Agent(
            # Config comes from external YAML/JSON
            config=self.agents_config['researcher'],
            verbose=True,
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
        """Reporting agent that compiles research findings into a structured report."""
        return Agent(
            # Config from external YAML/JSON
            config=self.agents_config['reporting_analyst'],
            verbose=True
        )

    # -------------------
    # Tasks
    # -------------------

    @task
    def research_task(self) -> Task:
        """Task definition for conducting research and producing structured output."""
        return Task(
            config=self.tasks_config['research_task'],  # Config from YAML/JSON
            output_json=ResearchOutput                 # Structured output model
        )

    @task
    def reporting_task(self) -> Task:
        """Task definition for analyzing research findings and writing a report."""
        return Task(
            # Config from YAML/JSON
            config=self.tasks_config['reporting_task'],
            output_json=ExecutiveReport                  # Structured output model
        )

    # -------------------
    # Crew
    # -------------------

    @crew
    def crew(self) -> Crew:
        """Assembles the DecisionCrew with its agents and tasks."""
        return Crew(
            agents=self.agents,  # Agents created by @agent decorators
            tasks=self.tasks,    # Tasks created by @task decorators
            process=Process.sequential,  # Tasks executed sequentially
            verbose=True,
            # Alternative: Process.hierarchical for different task orchestration
        )
