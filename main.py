import os
import argparse
import json
from typing import list, dict
import uuid
from crewai import Crew, Process

from MultiAgent.agents import CustomAgents
from MultiAgent.tasks import (
    search_task, upload_task, process_task, classification_task, 
    summary_task, synthesis_task, audio_task
)

os.makedirs("uploads", exist_ok=True)
os.makedirs("audio_files", exist_ok=True)
os.makedirs("results", exist_ok=True)

class ResearchPaperSummarizer:
    def __init__(self):
        ####Agents initialization
        self.agents = CustomAgents()
        agent_instances = {
            "research": self.agents.research_agent(),
            "processing": self.agents.processing_agent(),
            "classification": self.agents.classification_agent(),
            "summarization": self.agents.summarization_agent(),
            "synthesis": self.agents.synthesis_agent(),
            "audio": self.agents.audio_agent()
        }
        self.tasks = self._setup_tasks(agent_instances)
        
        ##### Setup the crew
        self.crew = Crew(
            agents=list(agent_instances.values()),
            tasks=self.tasks,
            process=Process.sequential,
            memory=True,
            verbose=True
        )
    
    def _setup_tasks(self, agents):
        ##### Linking tasks to agents
        tasks = [
            search_task.replace(agent=agents["research"]),
            upload_task.replace(agent=agents["processing"])
        ]
        
        process = process_task.replace(
            agent=agents["processing"],
            context=tasks[:2]  # search and upload
        )
        tasks.append(process)
        
        # Add classification task
        classify = classification_task.replace(
            agent=agents["classification"],
            context=[process]
        )
        tasks.append(classify)
        
        ##### Add summary task
        summarize = summary_task.replace(
            agent=agents["summarization"],
            context=[classify]
        )
        tasks.append(summarize)
        
        #### Add synthesis task
        synthesize = synthesis_task.replace(
            agent=agents["synthesis"],
            context=[summarize]
        )
        tasks.append(synthesize)
        
        #### Add audio task
        audio = audio_task.replace(
            agent=agents["audio"],
            context=[summarize, synthesize]
        )
        tasks.append(audio)
        
        return tasks
    
    def process_query(self, query=None, topics=None, max_results=5, 
                     pdf_paths=None, urls=None, dois=None):
        ##### Create session ID
        session_id = str(uuid.uuid4())[:8]
        result_dir = os.path.join("results", session_id)
        os.makedirs(result_dir, exist_ok=True)
        
        # Prepare inputs
        inputs = {
            'query': query,
            'topics': topics or ["AI", "Machine Learning", "NLP","Deep Learning", "Multi Agent"],
            'max_results': max_results,
            'session_id': session_id,
            'pdf_paths': pdf_paths or [],
            'urls': urls or [],
            'dois': dois or []
        }
        
        # Execute the crew and save the results
        results = self.crew.kickoff(inputs=inputs)
        self._save_results(results, result_dir)
        
        # Print summary
        print(f"\n{'='*50}\nResults saved to: {result_dir}")
        print(f"Papers: {len(results.get('papers', []))}")
        print(f"Summaries: {len(results.get('summaries', {}))}")
        print(f"Syntheses: {len(results.get('syntheses', {}))}")
        print(f"Audio files: {len(results.get('audio_files', []))}")
        
        return results
    
    def _save_results(self, results, result_dir):
        with open(os.path.join(result_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2) ####Save the json
        
        if "summaries" in results:
            summary_dir = os.path.join(result_dir, "summaries")
            os.makedirs(summary_dir, exist_ok=True)
            for paper_id, summary in results["summaries"].items():
                with open(os.path.join(summary_dir, f"{paper_id}.md"), "w") as f:
                    f.write(summary) #### Save the summaries
        
        if "syntheses" in results:
            synthesis_dir = os.path.join(result_dir, "syntheses")
            os.makedirs(synthesis_dir, exist_ok=True)
            for topic, synthesis in results["syntheses"].items():
                topic_file = topic.lower().replace(" ", "_") + ".md"
                with open(os.path.join(synthesis_dir, topic_file), "w") as f:
                    f.write(synthesis) #### Save the syntheses

# Command-line interface
def parse_args():
    parser = argparse.ArgumentParser(description="Research Paper Summarization System")
    parser.add_argument("--query", type=str, help="Search query")
    parser.add_argument("--topics", type=str, nargs="+", help="Topics for classification")
    parser.add_argument("--max-results", type=int, default=5, help="Max results per source")
    parser.add_argument("--pdf", type=str, nargs="*", help="PDF file paths")
    parser.add_argument("--url", type=str, nargs="*", help="Paper URLs")
    parser.add_argument("--doi", type=str, nargs="*", help="DOI references")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Validate input sources
    if not any([args.query, args.pdf, args.url, args.doi]):
        print("Error: Provide at least one input source (query, PDF, URL, or DOI)")
        exit(1)
    
    # Run system
    system = ResearchPaperSummarizer()
    system.process_query(
        query=args.query,
        topics=args.topics,
        max_results=args.max_results,
        pdf_paths=args.pdf,
        urls=args.url,
        dois=args.doi
    )
