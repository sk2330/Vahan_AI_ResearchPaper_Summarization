from crewai_tools import tool
import os
from transformers import pipeline
import requests
from bs4 import BeautifulSoup
import PyPDF2
import arxiv
from semanticscholar import SemanticScholar
import crossref_commons.retrieval
from gtts import gTTS
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import tempfile
from typing import List, Dict, Union,str
import uuid


@tool("pdf_text_extractor")
def pdf_text_extractor(file_path: str) -> str:
    """Extracts text from PDF files."""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            return '\n'.join([page.extract_text() for page in reader.pages])
    except Exception as e:
        raise RuntimeError(f"Error extracting PDF text: {e}")

@tool("arxiv_search")
def arxiv_search(query: str, max_results: int = 5, sort_by: str = "relevance") -> List[Dict]:
    """Search Arxiv for research papers."""
    sort_criteria = {
        "relevance": arxiv.SortCriterion.Relevance,
        "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
        "submittedDate": arxiv.SortCriterion.SubmittedDate,
    }
    sort_criterion = sort_criteria.get(sort_by, arxiv.SortCriterion.Relevance)
    
    search = arxiv.Search(query=query, max_results=max_results, sort_by=sort_criterion)
    return [
        {
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "summary": result.summary,
            "published": str(result.published),
            "url": result.pdf_url,
            "doi": getattr(result, 'doi', None),
            "source": "arXiv",
        }
        for result in search.results()
    ]

@tool("semantic_scholar_search")
def semantic_scholar_search(query: str, max_results: int = 5) -> List[Dict]:
    """Search Semantic Scholar for papers."""
    try:
        sch = SemanticScholar()
        results = sch.search_paper(query, limit=max_results)
        return [
            {
                "title": result.get("title", "Unknown Title"),
                "authors": [author.get("name", "Unknown Author") for author in result.get("authors", [])],
                "abstract": result.get("abstract", ""),
                "year": result.get("year", ""),
                "url": result.get("url", ""),
                "doi": result.get("externalIds", {}).get("DOI", ""),
                "source": "Semantic Scholar",
            }
            for result in results
        ]
    except Exception as e:
        raise RuntimeError(f"Error searching Semantic Scholar: {e}")

@tool("url_processor")
def url_processor(url: str) -> Dict[str, Union[str, Dict]]:
    """Process research papers from URLs (PDF or HTML)."""
    try:
        response = requests.get(url, stream=True, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        
        content_type = response.headers.get('Content-Type', '').lower()
        
        if 'application/pdf' in content_type or url.lower().endswith('.pdf'):
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                for chunk in response.iter_content(chunk_size=8192):
                    tmp.write(chunk)
                tmp_path = tmp.name
            
            text = pdf_text_extractor(tmp_path)
            os.unlink(tmp_path)  # Cleanup temporary file
            
            filename = os.path.basename(url)
            title = filename.replace('.pdf', '').replace('_', ' ').replace('-', ' ')
            
            return {"text": text, "metadata": {"title": title, "url": url, "source": "URL (PDF)"}}
        
        else:
            soup = BeautifulSoup(response.text, 'html.parser')
            for element in soup(["script", "style"]):
                element.extract()
            
            text = soup.get_text(separator='\n').strip()
            title = soup.title.string if soup.title else "Unknown Title"
            
            meta_tags = {meta['name'].lower(): meta['content'] for meta in soup.find_all('meta') if meta.get('name') and meta.get('content')}
            
            return {
                "text": text,
                "metadata": {
                    "title": title,
                    "authors": meta_tags.get('author', '').split(','),
                    "description": meta_tags.get('description', ''),
                    "url": url,
                    "source": "URL (HTML)",
                },
            }
    except Exception as e:
        raise RuntimeError(f"Error processing URL: {e}")

@tool("doi_resolver")
def doi_resolver(doi: str) -> Dict[str, Union[str, List]]:
    """Resolve DOI to retrieve paper metadata and URL."""
    try:
        doi = doi.strip().removeprefix('doi:').removeprefix('https://doi.org/')
        
        work = crossref_commons.retrieval.get_publication_as_json(doi)
        
        title = work.get('title', ['Unknown Title'])[0]
        authors = [' '.join(filter(None, [author.get('given'), author.get('family')])) for author in work.get('author', [])]
        
        abstract = work.get('abstract', '')
        year = work['published']['date-parts'][0][0] if 'published' in work and 'date-parts' in work['published'] else ''
        
        journal = work.get('container-title', [''])[0]
        publisher = work.get('publisher', '')
        
        url = f"https://doi.org/{doi}"
        
        return {
            "doi": doi,
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "year": year,
            "journal": journal,
            "publisher": publisher,
            "url": url,
            "source": "DOI",
        }
    except Exception as e:
        raise RuntimeError(f"Error resolving DOI: {e}")

@tool("text_summarizer")
def text_summarizer(text: str) -> str:
    """Summarizes text using Hugging Face's T5 model."""
    summarizer = pipeline("summarization",  model="google-t5/t5-small")
    summary = summarizer(text[:1024], max_length=500, min_length=50, do_sample=False)
    return summary[0]['summary_text']

@tool("topic_classifier")
def topic_classifier(text: str, topics: List[str]) -> Dict[str, float]:
    """Classifies text into topics using SentenceTransformer."""
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    text_embedding = model.encode(text)
    topic_embeddings = model.encode(topics)
    similarities = cosine_similarity([text_embedding], topic_embeddings)[0]
    return {topic: float(score) for topic, score in zip(topics, similarities)}

@tool("cross_paper_synthesizer")
def cross_paper_synthesizer(papers: list, topic: str) -> str:
    """Generates synthesis across multiple papers using Hugging Face's Flan-T5 model."""
    synthesizer = pipeline("text-generation", model="google/flan-t5-large")
    context = "\n\n".join([f"Title: {p['title']}\nSummary: {p['summary']}" for p in papers])
    prompt = f"Create a synthesis on the topic '{topic}' from the following papers:\n\n{context}"
    synthesis = synthesizer(prompt, max_length=500)
    return synthesis[0]['generated_text']

@tool("audio_generator")
def audio_generator(text: str) -> str:
    """Converts text to speech using gTTS."""
    filename = f"audio_{uuid.uuid4().hex[:8]}.mp3"
    filepath = os.path.join("audio_files", filename)
    os.makedirs("audio_files", exist_ok=True)
    gTTS(text=text[:5000], lang='en').save(filepath)
    return filepath

