# CHAIN OF TABLE LANGGRAPH

## Overview
This repository that demonstrates [Chain-of-Table](https://arxiv.org/abs/2401.04398) reasoning on multiple tables, which is SOTA research by folks at Google and Stanford powered by [LangGraph](https://github.com/langchain-ai/langgraph).

## Getting started

Follow these steps to set up:

1. **Clone the Repository**
    ```
    git clone https://github.com/aaliyan1230/chain-of-table-langgraph.git
    ```
2. **Configure environment**
    Navigate to the repository directory and run:
    ```
    python -m venv venv
    ```
    ```
    source venv/bin/activate
    ```
3. **Install Dependencies**
    Navigate to the repository directory and run:
    ```
    pip install -r requirements.txt
    ```
4. **Configure API Keys**
    Create a `.env` file in the root directory. Add your OpenAI API key and LangChain API details as follows, if you don't require logging via langsmith, only openai key would do fine too:
    ```
    OPENAI_API_KEY="..."
    LANGCHAIN_API_KEY="..."
    LANGCHAIN_TRACING_V2="..."
    LANGCHAIN_ENDPOINT="..."
    LANGCHAIN_PROJECT="..."
    ```

## Usage
TBD

## Data
TBD
