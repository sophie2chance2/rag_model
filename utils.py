import torch
import os
import bs4
import json
import numpy as np
import time

from pprint import pprint

import locale

from transformers import AutoTokenizer , AutoModelForCausalLM
from transformers import pipeline, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline
from langchain.llms import HuggingFacePipeline
from langchain_cohere import ChatCohere
from langchain import PromptTemplate, LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import Qdrant
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.utils.math import cosine_similarity

from langchain_community.document_loaders import ArxivLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.document_loaders import OnlinePDFLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import PubMedLoader

from langchain_qdrant import FastEmbedSparse, RetrievalMode


import pandas as pd
from datasets import Dataset
from sentence_transformers import CrossEncoder
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    answer_similarity,
    answer_correctness,
)

import dotenv
dotenv.load_dotenv()

locale.getpreferredencoding = lambda: "UTF-8"

COHERE_API_KEY = os.getenv('COHERE_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

# Retrieve the documents
def retrieve_documents(query, top_k=10):
    retrieved_docs = retriever.invoke(query, search_type="mmr", search_kwargs={"score_threshold": 0.5, "k":top_k})

    docs_with_sources = []
    for doc in retrieved_docs:
        # Combine summary and page content to get the most out of every chunk
        full_content = doc.metadata.get('summary', '') + " " + doc.page_content if 'summary' in doc.metadata else doc.page_content
        full_content = full_content.strip()
        
        docs_with_sources.append({
            'content': full_content,
            'source': doc.metadata.get('source', 'Unknown')
        })

    return docs_with_sources


def alternate_retrieval(query, top_k=15):
    retrieved_docs = retriever.invoke(query, search_type="similarity_score_threshold", search_kwargs={"k":top_k})

    docs_with_sources = []
    for doc in retrieved_docs:
        # Combine summary and page content to get the most out of every chunk
        full_content = doc.metadata.get('summary', '') + " " + doc.page_content if 'summary' in doc.metadata else doc.page_content
        full_content = full_content.strip()
        
        docs_with_sources.append({
            'content': full_content,
            'source': doc.metadata.get('source', 'Unknown')
        })

    return docs_with_sources