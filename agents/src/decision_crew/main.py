#!/usr/bin/env python
import json
import sys
import warnings

from datetime import datetime

from agents.src.decision_crew.crew import DecisionCrew

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information


def run(query) -> str:
    """
    Run the crew.
    """
    inputs = {
        "query": query,
    }

    try:
        result = DecisionCrew().crew().kickoff(inputs=inputs)
        return result.json_dict
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")
