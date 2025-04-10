import os
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

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('audio_files', exist_ok=True)
os.makedirs('results', exist_ok=True)

class ResearchPaperSummarizer:
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
        
        # Task setup
        self.tasks = self._setup_tasks(agent_instances)
        
        # Crew initialization
        self.crew = Crew(
            agents=list(agent_instances.values()),
            tasks=self.tasks,
            process=Process.sequential,
            memory=True,
            verbose=True
        )

    def _setup_tasks(self, agents):
        tasks = [
            search_task.replace(agent=agents["research"]),
            upload_task.replace(agent=agents["processing"])
        ]
        
        process = process_task.replace(
            agent=agents["processing"],
            context=tasks[:2]
        )
        tasks.append(process)
        
        tasks += [
            classification_task.replace(agent=agents["classification"], context=[process]),
            summary_task.replace(agent=agents["summarization"], context=[process]),
            synthesis_task.replace(agent=agents["synthesis"], context=[process]),
            audio_task.replace(agent=agents["audio"], context=[process])
        ]
        
        return tasks

    def process_query(self, query, topics, pdf_paths, urls, dois):
        session_id = str(uuid.uuid4())[:8]
        result_dir = os.path.join("results", session_id)
        os.makedirs(result_dir, exist_ok=True)

        inputs = {
            'query': query,
            'topics': topics,
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

# Initialize the summarizer system
summarizer = ResearchPaperSummarizer()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle form submission
        query = request.form.get('query', '')
        topics = request.form.get('topics', '').split(',')
        urls = request.form.get('urls', '').split(',')
        dois = request.form.get('dois', '').split(',')
        
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
