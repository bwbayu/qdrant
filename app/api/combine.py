from agents.src.decision_crew import main as decision_crew_main


def summary_generation(query="what is different about software engineer and software engineering"):
    result = decision_crew_main.run(query)
    return result


def video_link_generation():
    return [
        # ("https://samplelib.com/lib/preview/mp4/sample-5s.mp4", "Video pertama"),
        ("https://samplelib.com/lib/preview/mp4/sample-10s.mp4", "Video kedua")
    ]
# backend code
