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

## HuggingFaceEmbeddings
base_embeddings = HuggingFaceEmbeddings(model_name="multi-qa-mpnet-base-dot-v1")

## Mistral LLM without GPU
# llm_mistral_model = AutoModelForCausalLM.from_pretrained(
#     "mistralai/Mistral-7B-Instruct-v0.2",
#     torch_dtype=torch.float32,
#     device_map='cpu'  # Ensure it loads on CPU
# )
# llm_mistral_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

# mistral_pipe = pipeline(
#     "text-generation",
#     model=llm_mistral_model,
#     tokenizer=llm_mistral_tokenizer,
#     max_new_tokens=1000,
#     temperature=0.6,
#     top_p=0.95,
#     do_sample=True,
#     repetition_penalty=1.2
# )
# mistral_pipe.model.config.pad_token_id = mistral_pipe.model.config.eos_token_id
# mistral_llm_lc = HuggingFacePipeline(pipeline=mistral_pipe)

# # Cohere LLM
# cohere_chat_model = ChatCohere(cohere_api_key=COHERE_API_KEY)

################################## START OF LOADING DATA #########################################

## Store Data
CHUNK_SIZE=512
OVERLAP=150

text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP)

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
documents = loader.load()
splits = text_splitter.split_documents(documents)

qdrant_vectorstore = Qdrant.from_documents(splits,
    embedding = base_embeddings,
    location=":memory:",  # Local mode with in-memory storage only
    collection_name="rag_tech_db",
    force_recreate=True,
    retrieval_mode=RetrievalMode.DENSE,
    enable_limit=True
)

retriever = qdrant_vectorstore.as_retriever()
# #assign a unique number to each document we ingest
# global_doc_number = 1
# arxiv_numbers = ('2005.11401', '2104.07567', '2104.09864', '2105.03011', '2106.09685', '2203.02155', '2211.09260', '2211.12561',
#                  '2212.09741', '2305.14314', '2305.18290', '2306.15595', '2309.08872', '2309.15217', '2310.06825', '2310.11511',
#                  '2311.08377', '2312.05708', '2401.06532', '2401.17268', '2402.01306', '2402.19473', '2406.04744')

# all_arxiv_pages = []

# #loop through the papers
# for identifier in arxiv_numbers:
#     # Construct URL using the arXiv unique identifier
#     arx_url = f"https://arxiv.org/pdf/{identifier}.pdf"

#     # Extract pages from the document and add them to the list of pages
#     arx_loader = PyMuPDFLoader(arx_url)
#     arx_pages = arx_loader.load()
#     for page_num in range(len(arx_pages)):
#         page = arx_pages[page_num]
#         #CHANGED
#         page.metadata['page_num'] = page_num
#         page.metadata['doc_num'] = global_doc_number
#         page.metadata['doc_source'] = "ArXiv"
#         all_arxiv_pages.append(page)


#     global_doc_number += 1

# num_pages = len(all_arxiv_pages)
# num_docs = global_doc_number - 1

# #index doc chunks
# splits = text_splitter.split_documents(all_arxiv_pages)
# for idx, text in enumerate(splits):
#     splits[idx].metadata['split_id'] = idx

# qdrant_vectorstore.add_documents(documents=splits)

# wiki_docs = WikipediaLoader(query="Generative Artificial Intelligence", load_max_docs=4).load()
# for idx, text in enumerate(wiki_docs):
#     wiki_docs[idx].metadata['doc_num'] = global_doc_number
#     wiki_docs[idx].metadata['doc_source'] = "Wikipedia"

# global_doc_number += 1

# print('Number of documents: ', len(wiki_docs))

# #index docs
# wiki_splits = text_splitter.split_documents(wiki_docs)
# for idx, text in enumerate(wiki_splits):
#     wiki_splits[idx].metadata['split_id'] = idx

# print('Number of splits/chunks: ', len(wiki_splits))
# qdrant_vectorstore.add_documents(documents=wiki_splits)

# wiki_docs = WikipediaLoader(query="Information Retrieval", load_max_docs=4).load()
# for idx, text in enumerate(wiki_docs):
#     wiki_docs[idx].metadata['doc_num'] = global_doc_number
#     wiki_docs[idx].metadata['doc_source'] = "Wikipedia"

# global_doc_number += 1

# print('Number of documents: ', len(wiki_docs))

# #index docs
# wiki_splits = text_splitter.split_documents(wiki_docs)
# for idx, text in enumerate(wiki_splits):
#     wiki_splits[idx].metadata['split_id'] = idx

# print('Number of splits/chunks: ', len(wiki_splits))

# qdrant_vectorstore.add_documents(documents=wiki_splits)

# wiki_docs = WikipediaLoader(query="Large Language Models", load_max_docs=4).load()
# for idx, text in enumerate(wiki_docs):
#     wiki_docs[idx].metadata['doc_num'] = global_doc_number
#     wiki_docs[idx].metadata['doc_source'] = "Wikipedia"

# global_doc_number += 1

# print('Number of documents: ', len(wiki_docs))

# #index docs
# wiki_splits = text_splitter.split_documents(wiki_docs)
# for idx, text in enumerate(wiki_splits):
#     wiki_splits[idx].metadata['split_id'] = idx

# print('Number of splits/chunks: ', len(wiki_splits))

# qdrant_vectorstore.add_documents(documents=wiki_splits)

# web_loader = WebBaseLoader(
#     web_paths=("https://lilianweng.github.io/posts/2020-10-29-odqa/",
#                "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
#                "https://lilianweng.github.io/posts/2018-06-24-attention/",
#                "https://lilianweng.github.io/posts/2023-06-23-agent/",
#                "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/"),

#     bs_kwargs=dict(
#         parse_only=bs4.SoupStrainer(
#             class_=("post-content", "post-title", "post-header")
#         )
#     ),
# )

# web_documents = web_loader.load()

# for idx, text in enumerate(web_documents):
#     web_documents[idx].metadata['doc_num'] = global_doc_number
#     web_documents[idx].metadata['doc_source'] = "WWW"
# global_doc_number += 1

# print('Number of documents: ', len(web_documents))

# web_splits = text_splitter.split_documents(web_documents)

# for idx, text in enumerate(web_splits):
#     web_splits[idx].metadata['split_id'] = idx

# print('Number of splits: ', len(web_splits))

# qdrant_vectorstore.add_documents(documents=web_splits)

########################################### END OF LOADING DATA ###########################################

cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')

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

def rerank_documents(query, documents):
    cross_inp = [[query, doc['content']] for doc in documents]
    cross_scores = cross_encoder.predict(cross_inp)
    
    # Normalize scores to a 0-1
    min_score = np.min(cross_scores)
    max_score = np.max(cross_scores)
    normalized_scores = (cross_scores - min_score) / (max_score - min_score)
    
    # Sort documents by their normalized scores
    reranked_docs = [doc['content'] for doc in sorted(documents, key=lambda doc: normalized_scores[documents.index(doc)], reverse=True)]
    content_sources = [doc['source'] for doc in sorted(documents, key=lambda doc: normalized_scores[documents.index(doc)], reverse=True)]
    # scores = [score for score in sorted(cross_scores, reverse=True)]

    return reranked_docs, content_sources

def enhanced_retrieval_chain(inputs, top_k=5, sources = False):  
    query = inputs['question']

    # Dense Retrieval
    retrieved_docs = retrieve_documents(query, top_k=top_k*2)  # Retrieve more for re-ranking
    
    # Re-rank Retrieved Documents
    reranked_docs, content_sources = rerank_documents(query, retrieved_docs) #[:top_k]

    if sources:
        return reranked_docs, content_sources
    else:
        return reranked_docs

def alternate_retrieval_chain(inputs, top_k=5):  
    query = inputs['question']

    # Dense Retrieval - done using similarity this type and getting more values
    retrieved_docs = alternate_retrieval(query, top_k=top_k*3) 

    # Re-rank Retrieved Documents - still returning the same amount of info
    reranked_docs = rerank_documents(query, retrieved_docs)

    return reranked_docs

def rag_chain_call(question, user_type="marketing", model="mistral"):
    if user_type == "marketing":
        marketing_rag_template = """[INST]You are an expert marketing consultant hired to provide clear, concise, and engaging answers to complex questions for the marketing team. Please answer the question below based solely on the provided context. Ensure your response is brief, highlights the practical benefits, and is easy to understand.\n\nHere is a context:\n{context} \n\nHere is a question given by a marketing professional: \n{question}. In no more than 2 sentences, the answer to the question using the given context is [/INST]"""
        rag_prompt = ChatPromptTemplate.from_template(marketing_rag_template)
    elif user_type == "research":
        research_rag_template = """[INST]You are an expert technical consultant hired to provide clear, precise, and detailed answers to complex questions for the engineering team. Please answer the question below based solely on the provided context. Ensure your response is technical, highlights key functionalities and potential challenges, and is easy to understand for an engineering audience.\n\nHere is a context:\n{context} \n\nHere is a question given by an engineer: \n{question}. In no more than 2 sentences, the answer to the question using the given context is [/INST]""" 
        rag_prompt = ChatPromptTemplate.from_template(research_rag_template)
    else:
        other_rag_template = "[INST]You are an expert consultant hired to provide clear, precise, and engaging answers to complex questions for the " + user_type + " team. Please answer the question below based solely on the provided context. Ensure your response is brief, highlights key elements and potential challenges, and is easy to understand for a " + user_type + " audience.\n\nHere is a context:\n{context} \n\nHere is a question given by a " + user_type +" professional: \n{question}. In no more than 2 sentences, the answer to the question using the given context is [/INST]"
        rag_prompt = ChatPromptTemplate.from_template(other_rag_template)
    
    if model == "mistral":
        rag_chain = (
            {"context": enhanced_retrieval_chain,
             "question": RunnablePassthrough()}
            | rag_prompt
            | mistral_llm_lc
        )
        response = rag_chain.invoke({"question": question})
        response = response.split('[/INST]')[-1].strip()

    elif model == "cohere":
        rag_chain = (
            {"context": enhanced_retrieval_chain,
             "question": RunnablePassthrough()}
            | rag_prompt
            | cohere_chat_model
        )
        response = rag_chain.invoke({"question": question})
        response = response.content.replace("[/INST]", "")
    else:
        raise ValueError("Unsupported model specified")
    
    return response

def rag_chain_call_second_try(question, original_response, user_type="marketing", model="mistral"):
    if user_type == "marketing":
        marketing_rag_template = """[INST]You are an expert marketing consultant hired to provide clear, concise, and engaging answers to complex questions for the marketing team. Please answer the question below based solely on the provided context. Ensure your response is brief, highlights the practical benefits, and is easy to understand.\n\nHere is a context:\n{context} \n\nHere is a question given by a marketing professional: \n{question}\n\nHere is the original response: \n{original_response}\n\nYour last answer gave incorrect information. If this one is incorrect, many people will lose their jobs. In no more than 2 sentences, the CORRECT answer to the question using the given context is [/INST]"""
        rag_prompt = ChatPromptTemplate.from_template(marketing_rag_template)
    elif user_type == "research":
        research_rag_template = """[INST]You are an expert technical consultant hired to provide clear, precise, and detailed answers to complex questions for the engineering team. Please answer the question below based solely on the provided context. Ensure your response is technical, highlights key functionalities and potential challenges, and is easy to understand for an engineering audience.\n\nHere is a context:\n{context} \n\nHere is a question given by an engineer: \n{question}\n\nHere is the original response: \n{original_response}\n\nYour last answer gave incorrect information. If this one is incorrect, many people will lose their jobs. In no more than 2 sentences, the CORRECT answer to the question using the given context is [/INST]""" 
        rag_prompt = ChatPromptTemplate.from_template(research_rag_template)
    else:
        other_rag_template = "[INST]You are an expert consultant hired to provide clear, precise, and engaging answers to complex questions for the " + user_type + " team. Please answer the question below based solely on the provided context. Ensure your response is brief, highlights key elements and potential challenges, and is easy to understand for a " + user_type + " audience.\n\nHere is a context:\n{context} \n\nHere is a question given by a " + user_type +" professional: \n{question}\n\nHere is the original response: \n{original_response}\n\n Your last answer gave incorrect information. If this one is incorrect, many people will lose their jobs. In no more than 2 sentences, the CORRECT answer to the question using the given context is [/INST]"
        rag_prompt = ChatPromptTemplate.from_template(other_rag_template)
    
    if model == "mistral":
        rag_chain = (
            {"context": alternate_retrieval_chain,
             "question": RunnablePassthrough()}
            | rag_prompt
            | mistral_llm_lc
        )
        response = rag_chain.invoke({"question": question, "original_response":original_response})
        response = response.split('[/INST]')[-1].strip()

    elif model == "cohere":
        rag_chain = (
            {"context": alternate_retrieval_chain,
             "question": RunnablePassthrough()}
            | rag_prompt
            | cohere_chat_model
        )
        response = rag_chain.invoke({"question": question, "original_response":original_response})
        response = response.content.replace("[/INST]", "")
    else:
        raise ValueError("Unsupported model specified")
    
    return response

def get_context(response):
    context = response.split("\n\nHere is a context:")[-1].split('Here is a question given by a marketing professional:')[0].strip()
    return context

def get_answer(response):
    answer = response.split('[/INST]')[-1].strip()
    return answer

def get_true_false(fact_check_response):
    if 'true' in fact_check_response.lower():
        return 'True'
    elif 'false' in fact_check_response.lower():
        return 'False'
    else:
        return fact_check_response
    
def fact_checker(question, response, model="mistral"):
    fact_check_rag_template = """[INST]You are a diligent fact checker, use only the context given to determine if the claims given in the response are True or False. \n\nHere is the context:\n{context}\n\nHere is the question:\n{question}\n\nHere is the response:\n{response}\n\nGiven the context, this response is (True/False): [/INST]"""
    rag_prompt = ChatPromptTemplate.from_template(fact_check_rag_template)
    
    if model == "mistral":
        rag_chain = (
            {"context": enhanced_retrieval_chain, 
             "question": RunnablePassthrough(),
             "response": RunnablePassthrough()}
            | rag_prompt
            | mistral_llm_lc
        )
        response = rag_chain.invoke({"question": question, "response": response})
        response = response.split('[/INST]')[-1].strip()

    elif model == "cohere":
        rag_chain = (
            {"context": enhanced_retrieval_chain,
             "question": RunnablePassthrough(),
             "response": RunnablePassthrough()}
            | rag_prompt
            | cohere_chat_model
        )
        response = rag_chain.invoke({"question": question, "response": response})
        response = response.content.replace("[/INST]", "")
    else:
        raise ValueError("Unsupported model specified")

    true_false = get_true_false(response)
    
    return true_false

def run_rag_with_fact_check(question, department, initial_model="mistral"):
    # Answer
    response = rag_chain_call(question, user_type=department, model=initial_model)
    answer = get_answer(response)

    # Fact Check
    fact_check = fact_checker(question, answer, model="mistral")

    # When facts are wrong - try again, different retrieval method for context and new prompt
    if fact_check != 'True':
      response = rag_chain_call_second_try(question, answer, user_type=department, model="cohere")
      answer = get_answer(response)

    return answer, fact_check

##################### END DEFINING FUNCTIONS ############################

question = "What purpose do large language models serve in the field of natural language processing?"
# marketing_answer = rag_chain_call(question, "marketing", "mistral")

# marketing_answer, marketing_fact_check = run_rag_with_fact_check(question, "marketing", "mistral")
# research_answer, research_fact_check = run_rag_with_fact_check(question, "research", "mistral")

# context, sources = enhanced_retrieval_chain({"question": question}, top_k=5, sources = True)

def call_model(question, department, model):
    response = enhanced_retrieval_chain({"question": question}, top_k=5, sources = True) # just getting context for now since call is taking so long
    # response = rag_chain_call(question, department, model)
    return response
