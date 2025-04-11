### Setup Instructions

1. Clone the Repo / Download the zip file and extract it (For cloning use the GitCLI)
2. After Extracting, copy the folder path. 
3. Open the Anaconda Prompt and use the following commands:
1. cd your_file_path
2. D: / C:/ E: (in which ever drive it is present)
3. next use - code . (to open VS code in that location itself)

4. After opening, create a virtual environment to run the project:
1. conda create -p venv (your virtual environment name- can be anything) python== 3.10 (the version I used for this project)
2. conda activate venv/

5. Install the requirements and run the project!!
1. pip install -r requiremennts .txt
2. python main.py

### System Architecture

The system uses a multi-agent architecture to perform research paper summarization tasks:

1. **Agents**:
    - Research Agent: Searches for papers using queries, DOIs, or URLs.
    - Processing Agent: Extracts text from uploaded PDFs or online resources.
    - Classification Agent: Categorizes papers into user-defined topics.
    - Summarization Agent: Generates structured summaries of individual papers.
    - Synthesis Agent: Produces cross-paper syntheses highlighting common themes and gaps.
    - Audio Agent: Converts summaries and syntheses into audio files.

2. **Tasks**:
    Each agent is assigned specific tasks that contribute to the overall workflow.

3. **Tools**:
    Tools like PDF extractors, DOI resolvers, URL processors, and text summarizers are used by agents to complete tasks efficiently.

4. **Frontend**:
    A Flask-based web interface allows users to input queries, upload PDFs, and view/download results.

### Multi-Agent Design

The system uses CrewAI's multi-agent framework where agents collaborate as follows:

1. Agents work independently but communicate through shared memory (CrewAI's memory module).
2. Tasks are executed sequentially using CrewAI's 'Process.sequential'.
3. Each agent has specific tools (e.g., PDF extractor) to perform its role efficiently.

This approach ensures modularity, scalability, and flexibility in handling complex workflows.


### Paper Processing Methodology

1. Search Phase:
    - Queries are sent to arXiv or Semantic Scholar APIs to retrieve relevant papers.

2. Extraction Phase:
    - PDFs are processed using PyPDF2 to extract text content.
    - Metadata is extracted from DOIs using CrossRef API.

3. Classification Phase:
    - Papers are categorized into user-defined topics using SentenceTransformer embeddings.

4. Summarization Phase:
    - Summaries are generated using Hugging Face's T5 model.

5. Synthesis Phase:
    - Syntheses are created using Flan-T5 model by analyzing relationships between papers.

6. Audio Generation Phase:
    - Summaries/syntheses are converted into MP3 files using gTTS (Google Text-to-Speech).

### Audio Generation Implementation
Audio files are generated using gTTS (Google Text-to-Speech). The steps include:
1. Convert summary/synthesis text into speech using gTTS API.
2. Save speech as MP3 files in the 'audio_files' directory.
3. Provide download links for audio files in the web interface.

### Limitations

1. Input Sources:
    Currently supports arXiv/Semantic Scholar APIs but could expand to other repositories like IEEE Xplore.

2. Use Of Multiple Open Source Models:
    To optimize costs, multiple open-source models are utilized based on the specific task. However, this approach may result in a slight increase in execution time by a few seconds compared to standard operations

### Future Improvements
1. Integrating OpenAI/ Grok/ Perplexity Model:
    The utilization of APIs for accessing various models typically incurs a nominal cost. However, when integrated with CrewAI, the overall performance, efficiency, and speed of the models are significantly enhanced.


