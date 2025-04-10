import os
from dotenv import load_dotenv
import json
import uuid
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
from crewai import Crew, Process
from MultiAgent.agents import CustomAgents
from MultiAgent.tasks import (  
    search_task, upload_task, process_task,
    classification_task, summary_task,
    synthesis_task, audio_task
)

load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('audio_files', exist_ok=True)
os.makedirs('results', exist_ok=True)

class ResearchPaperSummarizer:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set via .env or parameter.")
            
        os.environ["OPENAI_API_KEY"] = self.api_key 

    def __init__(self):
        self.agents = CustomAgents()
        agent_instances = {
            "research": self.agents.research_agent(),
            "processing": self.agents.processing_agent(),
            "classification": self.agents.classification_agent(),
            "summarization": self.agents.summarization_agent(),
            "synthesis": self.agents.synthesis_agent(),
            "audio": self.agents.audio_agent()
        }
        
        self.tasks = self._map_tasks(agent_instances)
        
        self.crew = Crew(
            agents=list(agent_instances.values()),
            tasks=self.tasks,
            process=Process.sequential,
            memory=True,
            verbose=True
        )

    def _map_tasks(self, agents):
        """Dynamically map agents to imported tasks"""
        return [
            self._create_task(search_task, agents["research"]),
            self._create_task(upload_task, agents["processing"]),
            self._create_task(process_task, agents["processing"], 
                            context=[search_task, upload_task]),
            self._create_task(classification_task, agents["classification"], 
                            context=[process_task]),
            self._create_task(summary_task, agents["summarization"], 
                            context=[classification_task]),
            self._create_task(synthesis_task, agents["synthesis"], 
                            context=[summary_task]),
            self._create_task(audio_task, agents["audio"], 
                            context=[synthesis_task])
        ]

    def _create_task(self, base_task, agent, context=None):
        return type(base_task)(
            description=base_task.description,
            agent=agent,
            expected_output=base_task.expected_output,
            tools=base_task.tools.copy() if hasattr(base_task, 'tools') else [],
            context=context or [],
            allow_delegation=getattr(base_task, 'allow_delegation', False),
            verbose=getattr(base_task, 'verbose', False)
        )

    def process_query(self, query, topics, pdf_paths, urls, dois,max_results=5):
        session_id = str(uuid.uuid4())[:8]
        result_dir = os.path.join("results", session_id)
        os.makedirs(result_dir, exist_ok=True)

        inputs = {
            'query': query,
            'topics': topics,
            'max_results': max_results,
            'pdf_paths': pdf_paths,
            'urls': urls,
            'dois': dois,
            'session_id': session_id
        }

        results = self.crew.kickoff(inputs=inputs)
        self._save_results(results, result_dir)
        
        return {
            'session_id': session_id,
            'result_count': {
                'papers': len(results.get('papers', [])),
                'summaries': len(results.get('summaries', {})),
                'syntheses': len(results.get('syntheses', {})),
                'audio_files': len(results.get('audio_files', []))
            }
        }

    def _save_results(self, results, result_dir):
        with open(os.path.join(result_dir, "results.json"), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save individual files
        if 'summaries' in results:
            summary_dir = os.path.join(result_dir, 'summaries')
            os.makedirs(summary_dir, exist_ok=True)
            for paper_id, summary in results['summaries'].items():
                with open(os.path.join(summary_dir, f'{paper_id}.md'), 'w') as f:
                    f.write(summary)
        
        if 'syntheses' in results:
            synthesis_dir = os.path.join(result_dir, 'syntheses')
            os.makedirs(synthesis_dir, exist_ok=True)
            for topic, synthesis in results['syntheses'].items():
                filename = topic.lower().replace(' ', '_') + '.md'
                with open(os.path.join(synthesis_dir, filename), 'w') as f:
                    f.write(synthesis)

#### Initialize the summarizer system
summarizer = ResearchPaperSummarizer()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle form submission
        query = request.form.get('query', '')
        topics = [t.strip() for t in request.form.get('topics', '').split(',') if t.strip()]
        urls = [u.strip() for u in request.form.get('urls', '').split(',') if u.strip()]
        dois = [d.strip() for d in request.form.get('dois', '').split(',') if d.strip()]
        
        # Handle file uploads
        pdf_paths = []
        for file in request.files.getlist('pdf_files'):
            if file.filename != '':
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                pdf_paths.append(file_path)
        
        # Process the query
        result = summarizer.process_query(
            query=query,
            topics=topics,
            pdf_paths=pdf_paths,
            urls=urls,
            dois=dois
        )
        
        return render_template('results.html', result=result)
    
    return render_template('index.html')

@app.route('/results/<session_id>')
def show_results(session_id):
    result_file = os.path.join('results', session_id, 'results.json')
    if os.path.exists(result_file):
        with open(result_file) as f:
            results = json.load(f)
        return render_template('detailed_results.html', results=results)
    return "Results not found", 404

@app.route('/download/<session_id>/<file_type>/<filename>')
def download_file(session_id, file_type, filename):
    valid_types = ['summaries', 'syntheses', 'audio']
    if file_type not in valid_types:
        return "Invalid file type", 400
    
    directory = os.path.join('results', session_id, file_type)
    return send_from_directory(directory, filename)

if __name__ == '__main__':
    app.run(debug=True)
