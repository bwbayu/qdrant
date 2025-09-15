from agents.src.decision_crew import main as decision_crew_main


def summary_generation(query="arsitektur dan organisasi komputer"):
    result = decision_crew_main.run(query)
    return result


# backend code
