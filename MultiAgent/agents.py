from crewai import Agent
from textwrap import dedent
from transformers import pipeline
from sentence_transformers import SentenceTransformer

class CustomAgents:
    def __init__(self):
        # Initializing open-source models
        self.Summarizer = pipeline("summarization", model="google-t5/t5-small")
        self.TextGeneration = pipeline("text-generation", model="google/flan-t5-large")
        self.TopicClassifier = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def research_agent(self):
        return Agent(
            role="Research Paper Collector",
            backstory=dedent("""
                You are an expert academic researcher skilled in navigating databases 
                like arXiv and Semantic Scholar to find relevant papers on specific topics.
            """),
            goal=dedent("""
                Search for and collect research papers based on user queries, 
                including metadata like title, authors, abstract, and publication year.
            """),
            tools=["arxiv_search", "semantic_scholar_search", "doi_resolver"],
            allow_delegation=False,
            verbose=True
        )

    def processing_agent(self):
        return Agent(
            role="Document Processing Specialist",
            backstory=dedent("""
                You specialize in extracting clean text and metadata from documents 
                in various formats, such as PDFs and HTML pages.
            """),
            goal=dedent("""
                Process research papers to extract structured text and metadata 
                for further analysis.
            """),
            tools=["pdf_text_extractor", "url_processor"],
            allow_delegation=False,
            verbose=True
        )

    def classification_agent(self):
        return Agent(
            role="Topic Classification Expert",
            backstory=dedent("""
                You are a subject matter expert with deep knowledge across multiple 
                academic disciplines, skilled in classifying papers into relevant topics.
            """),
            goal=dedent("""
                Analyze research paper content and classify it into user-defined topics 
                using semantic similarity scoring.
            """),
            tools=["topic_classifier"],
            allow_delegation=False,
            verbose=True,
            llm=self.TopicClassifier  # Use SentenceTransformer for classification
        )

    def summarization_agent(self):
        return Agent(
            role="Summarization Expert",
            backstory=dedent("""
                You excel at condensing complex research papers into concise summaries 
                while preserving key findings and methodologies.
            """),
            goal=dedent("""
                Generate structured summaries for individual research papers, focusing 
                on key findings, methodology, and conclusions. Include citations for each paper.
            """),
            tools=["text_summarizer", "citation_generator"],  # Added citation tool
            allow_delegation=False,
            verbose=True,
            llm=self.Summarizer  # Use T5 for summarization
        )

    def synthesis_agent(self):
        return Agent(
            role="Research Synthesizer",
            backstory=dedent("""
                You are a research director skilled at synthesizing insights from multiple 
                papers to identify common themes, contradictions, and future directions.
            """),
            goal=dedent("""
                Create a comprehensive synthesis across multiple research papers on a given topic, 
                highlighting common findings and identifying gaps in knowledge. Include citations for all referenced papers.
            """),
            tools=["cross_paper_synthesizer", "citation_generator"],  # Added citation tool
            allow_delegation=False,
            verbose=True,
            llm=self.TextGeneration  # Use Flan-T5 for synthesis
        )

    def audio_agent(self):
        return Agent(
            role="Audio Content Producer",
            backstory=dedent("""
                You specialize in converting academic content into engaging audio formats, 
                ensuring clarity and accessibility for listeners.
            """),
            goal=dedent("""
                Convert research paper summaries or syntheses into high-quality audio files 
                suitable for podcasts or presentations.
            """),
            tools=["audio_generator"],
            allow_delegation=False,
            verbose=True
        )
