# Virtual Assistant

## Overview
This project is a virtual assistant powered by GPT with additional functionalities, including text-to-speech, syntax highlighting, and real-time context processing. It is built with FastAPI and uses various libraries for NLP, speech synthesis, and scheduling tasks.

## Features
- **Text Processing & NLP**: Utilizes `TfidfVectorizer` and `cosine_similarity` for text analysis.
- **Text-to-Speech**: Uses `edge_tts` for converting text into speech.
- **Syntax Highlighting**: Implements `Pygments` for rendering code snippets.
- **Web API**: Built with `FastAPI`, supporting HTTP requests and WebSocket communication.
- **Scheduling Tasks**: Uses `APScheduler` for automated job scheduling.
- **Language Detection**: Uses `langdetect` to identify text language.
- **Markdown Rendering**: Implements `mistune` for Markdown processing.

## Installation
### Prerequisites
- Python 3.11.5

### Install Dependencies
```sh
pip install -r requirements.txt
```

## Usage
### Run the API Server
```sh
uvicorn run:app --host 0.0.0.0 --port 8000 --reload
```

### API Endpoints
- `GET /` - Home page
- `GET /synthesis` - Convert text to speech
- `GET /get_response` - Get response from assistant

## License
This project is licensed under the MIT License.

## Contact
For issues or contributions, feel free to open a GitHub issue or reach out via [Telegram @ikymuco](https://t.me/ikymuco).

