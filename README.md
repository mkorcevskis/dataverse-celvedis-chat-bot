# Dataverse Ceļvedis Chatbot
An intelligent question–answering assistant that uses the content of [the DataverseLV knowledge base](https://dataverse.lv/celvedis/) as a local knowledge base. The project demonstrates how to build a retrieval-augmented generation (RAG) pipeline using:
- [LangChain](https://www.langchain.com/)
- [Chroma](https://www.trychroma.com/)
- [Streamlit](https://streamlit.io/)
- [Ollama](https://ollama.com/)

## Features
- Interactive Q&A interface built on and run with Streamlit
- Local knowledge base built from the DataverseLV knowledge base
- LLM integration (via Ollama), developed and tested on "llama3.1:8b"
- Uses modern LangChain components
- Responses in Latvian, based solely on the indexed content

## Project Structure
```
DataverseCelvedisChatBot/
│
├── .idea/                  # PyCharm IDE configuration (excluded from Git)
│   └── ...
│
├── .streamlit/             # Streamlit theme and UI settings
│   └── config.toml
│
├── .venv/                  # Virtual environment (excluded from Git)
│   └── ...
│
├── storage/                # Prebuilt Chroma knowledge base (prebuilt, ready to use)
│   └── ...
│
├── .env                    # Environment variables (API keys, model names, paths)
├── .gitignore              # Git ignore rules
├── app.py                  # Streamlit application
├── index.py                # Script for building/rebuilding the knowledge base
├── LICENSE                 # License file
├── README.md               # Project documentation
├── requirements.txt        # Project dependencies
└── urls.txt                # List of URLs of webpages to index
```

## Setup
1. Clone the repository
2. Create and activate virtual environment (venv)
3. Install dependencies from the `requirements.txt` file
4. Download and install Ollama with an LLM you want to use, e.g. "llama3.1:8b"
5. Start the LLM with the `ollama serve` command
6. Run the Streamlit application with the `streamlit run app.py` command

**NB: By default, the project already includes a prebuilt knowledge base inside the `storage/` directory – you can use it right away without additional setup.**

**Optional:** If you want to recreate or update the knowledge base yourself, run the indexing script in the `index.py` file with the `python index.py` command. This script will:
- Read the list of URLs to be used as a knowledge base from the `urls.txt` file
- Fetch and process the content
- Generate embeddings
- Rebuild the Chroma database inside the `storage/` directory.

## License
This project is distributed under the MIT License.

## Acknowledgements
This project was developed within the project "Support for implementing Open Science, developing solutions for shared use of research data, and participation in the EOSC" (RRF Project No. 2.1.3.1.i) with the financial support of the European Union Recovery and Resilience Facility and the Republic of Latvia.
