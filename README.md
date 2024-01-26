# RAG Chatbot
Código do RAG desenvolvido para o post "Como criar um chatbot com seus próprios documentos usando LangChain e ChatGPT" do blog da DP6

## Como usar

Para usar o RAG, basta executar `python main.py` e conversar com o bot.

## Instalação

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install python-dotenv langchain langchain-openai openai milvus pymilvus unstructured tiktoken lark
```

## Configuração do ambiente

Criar arquivo `.env` com as seguintes variáveis:

```bash
OPENAI_API_KEY = ""
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
```