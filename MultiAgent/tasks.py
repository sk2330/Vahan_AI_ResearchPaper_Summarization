from crewai import Task
from MultiAgent.agents import CustomAgents

agents = CustomAgents()

research_agent = agents.research_agent()
processing_agent = agents.processing_agent()
classification_agent = agents.classification_agent()
summarization_agent = agents.summarization_agent()
synthesis_agent = agents.synthesis_agent()
audio_agent = agents.audio_agent()

search_task = Task(
    description="""
    Search for research papers based on the given query:
    1. Use arxiv_search and semantic_scholar_search to find relevant papers
    2. Apply filtering options (relevance, recency) as specified in the inputs
    3. For each paper, collect title, authors, abstract, publication year, and URL
    4. If DOI references are provided, resolve them using doi_resolver
    5. Return a complete list with at least 5 relevant papers with metadata
    
    Papers should be returned as a structured list with consistent metadata fields.
    """,
    expected_output="List of research papers with complete metadata in JSON format",
    agent=research_agent
)

upload_task = Task(
    description="""
    Process uploaded PDF research papers:
    1. Use pdf_text_extractor to extract text content from uploaded PDF files
    2. Parse the extracted text to identify document structure
    3. Extract metadata (title, authors, publication date) from the PDF
    4. Return the structured text content and metadata
    
    Output should be in the same format as papers from search results.
    """,
    expected_output="Dictionary of processed uploaded papers with text and metadata",
    agent=processing_agent
)

process_task = Task(
    description="""
    Process all papers from search results and uploads:
    1. For each paper, extract the full text using the appropriate method:
       - For PDFs, use pdf_text_extractor
       - For URLs, use url_processor to extract content
       - For DOIs, use the resolved content from doi_resolver
    2. Clean the extracted text by removing irrelevant elements and fixing formatting
    3. Extract and structure metadata (title, authors, publication date, journal, etc.)
    4. Organize text into sections (abstract, introduction, methodology, results, etc.)
    5. Return processed papers with structured text and complete metadata
    
    The output should preserve the logical structure of each paper for better analysis.
    """,
    expected_output="Dictionary of processed papers with extracted text and structured metadata",
    agent=processing_agent,
    context=[search_task, upload_task]
)


classification_task = Task(
    description="""
    Classify all processed papers according to the provided topic list:
    1. Use the topic_classifier tool to analyze each paper's content
    2. Calculate relevance scores for each topic (scale 0.0-1.0)
    3. Assign primary topic (highest score) and secondary topics (score > 0.5)
    4. Group papers by primary topic
    5. Return a comprehensive classification with papers organized by topic and relevance scores
    
    Input topics: {{topics}}
    
    Ensure papers that cover multiple topics are properly cross-referenced in the classification.
    """,
    expected_output="Dictionary mapping topics to relevant papers with confidence scores",
    agent=classification_agent,
    context=[process_task]
)

summary_task = Task(
    description="""
    Create comprehensive summaries for each processed paper:
    1. Use text_summarizer to generate concise yet complete summaries
    2. For each summary, include:
       - Main research question/objective
       - Key methodology components
       - Primary findings and results
       - Important conclusions and implications
    3. Generate a citation for each paper using citation_generator
    4. Format each summary with clear sections and logical structure
    5. Return all summaries with their corresponding paper metadata and citations
    
    Summaries should be concise but capture essential details that differentiate each paper.
    """,
    expected_output="Dictionary of paper summaries with citations and metadata",
    agent=summarization_agent,
    context=[classification_task]
)

synthesis_task = Task(
    description="""
    Generate cross-paper syntheses for each identified topic:
    1. For each topic from the classification results:
       a. Gather all papers primarily classified under that topic
       b. Use cross_paper_synthesizer to analyze relationship between papers
       c. Create a synthesis that:
          - Identifies common themes and consistent findings across papers
          - Highlights differences in methodologies and conflicting results
          - Summarizes the current state of knowledge on the topic
          - Identifies research gaps and future directions
    2. Include citations for all referenced papers using citation_generator
    3. Structure each synthesis with clear sections (overview, themes, contradictions, gaps)
    4. Return syntheses organized by topic, including all relevant paper references
    
    Topic syntheses should provide insights beyond individual papers, showing connections
    and patterns across the research landscape.
    """,
    expected_output="Dictionary of topic syntheses with cross-paper analysis and citations",
    agent=synthesis_agent,
    context=[summary_task]
)

audio_task = Task(
    description="""
    Convert all paper summaries and topic syntheses to audio format:
    1. For each paper summary:
       a. Format text for better audio consumption (spell out abbreviations, etc.)
       b. Generate an audio file using audio_generator
       c. Create a properly named audio file with paper title and authors
    
    2. For each topic synthesis:
       a. Format text for better spoken flow
       b. Generate an audio file using audio_generator
       c. Create a properly named audio file with topic name
    
    3. Return a structured list of audio files with:
       - File path
       - Type (summary or synthesis)
       - Associated paper title or topic
       - Duration
    
    Audio format should be clear, well-paced, and organized for listener comprehension.
    """,
    expected_output="Dictionary mapping paper/topic IDs to audio file paths with metadata",
    agent=audio_agent,
    context=[summary_task, synthesis_task]  # Including both summary and synthesis tasks
)
