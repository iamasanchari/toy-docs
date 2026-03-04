# toy-docs
Toy-docs is a project that explores the Capability to interact with a document or pdf file using LLMs

## Prerequisites
### Setup of olamma 
Install the olamma  https://ollama.com/download  and setup the LAMMA 3 (Configurable via config.py file )

### Packages to be installed


```bash
pip3 install langchain_community

pip3 install langchain_ollama      

pip3 install pypdf

pip3 install chromadb
```


## To seed the data

```bash
python3 ingest.py
```

## To run the CLI chat

```bash

python3 chat-cli.py
```

# Optional set python venv
```bash
python3 -m venv .venv
source .venv/bin/activate
python3 ingest.py  
```



