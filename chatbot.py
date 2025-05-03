from huggingface_hub import hf_hub_download

# Download the model
model_path = hf_hub_download(repo_id = "MaziyarPanahi/BioMistral-7B-GGUF", filename = "BioMistral-7B.Q4_K_M.gguf")

print(f"Model downloaded to: {model_path}")

"""## Installation"""

"""## Importing libraries"""

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_community.llms import LlamaCpp
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA, LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter


"""## Importing document"""

loader = PyPDFDirectoryLoader("C:\\Users\\twish\\OneDrive\\Desktop\\GestureConnect\\GC - Models\\Chatbot Dataset")
docs = loader.load()

len(docs)       # number of pages in pdf

docs[5]

"""## Chunking - dividing whole pdf into multiple parts to ease the process of putting the data into the LLM model."""

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 300, chunk_overlap = 50)
chunks = text_splitter.split_documents(docs)

len(chunks)

chunks[600]

chunks[892]

"""## Embeddings creations"""

import os

os.environ['HF_HOME'] = "C:\\Users\\twish\\.cache\\huggingface"

embeddings = SentenceTransformerEmbeddings(model_name = 'NeuML/pubmedbert-base-embeddings')

"""## Vector Store Creation"""

import shutil

shutil.rmtree("chroma_db", ignore_errors=True)

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)

# checking search

query = "How does yoga impact mental well-being?"

search_results = vector_store.similarity_search(query)

search_results

retriever = vector_store.as_retriever(search_kwargs = {'k': 5})    # top 5 results

retriever.get_relevant_documents(query)

"""## LLM Model Loading"""

llm = LlamaCpp(
    model_path="C:\\Users\\twish\\.cache\\huggingface\\hub\\models--MaziyarPanahi--BioMistral-7B-GGUF\\snapshots\\ae4f07544f1015dc8f5bf382b7582852638cbecf\\BioMistral-7B.Q4_K_M.gguf",
    temperature=0.2,
    max_tokens=4096,
    top_p=0.8,
    top_k=50,
    n_gpu_layers=-1,
    n_batch=512,
    n_threads=8,
    logits_all=True,
    f16_kv=True
)

import torch
print(torch.cuda.is_available())  # Should print True


"""## Using LLM and retriever and query to generate final response"""

template = '''
<|context|>
You are a Physical and Mental Well-being Assistant that follows the instructions and generate the accurate responses based on the query and context provided.
Please be truthful and give direct answers.
</s>
<|user|>
{query}
</s>
<|assistant|>
'''

from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever, "query": RunnablePassthrough()}
    | prompt                              # | = pipe, it indicates that the first step has taken place, now its the next step
    | llm
    | StrOutputParser()
)

response = rag_chain.invoke(query)                 # query = "How does yoga impact mental well-being?"

response

import sys

"""### Context Awareness"""

from random import uniform
import time
import matplotlib.pyplot as plt

# ✅ Store last 20 messages for context
conversation_history = []
response_times = []
confidence_scores = []
user_inputs = []
bot_outputs = []
timestamps = []

def update_conversation_history(user_input, bot_response=None):
    """Stores the last 20 messages for context."""
    context_window = 20
    conversation_history.append(f"User: {user_input}")
    if len(conversation_history) > context_window:
        conversation_history.pop(0)

    prompt = "\n".join(conversation_history)  # Pass full context to model
    print("Current Prompt:\n", prompt)

    start_time = time.time()
    end_time = time.time()

    # 🧠 Dynamic confidence (mocked for now)
    confidence = round(uniform(0.65, 0.95), 2)


    response = llm.generate(prompts=[user_input])
    conversation_history.append(f"Bot: {response}")
    timestamps.append(len(user_inputs))
    response_times.append(round(end_time - start_time, 2))
    confidence_scores.append(confidence)
    user_inputs.append(user_input)
    bot_outputs.append(response)

    print("Response:", response)
    print(f"Response Time: {response_times[-1]}s | Confidence: {confidence}")


def plot_conversation():
    plt.figure(figsize=(12, len(user_inputs) * 0.6))
    for i, (u, b) in enumerate(zip(user_inputs, bot_outputs)):
        plt.text(0.1, len(user_inputs) - i, f"🧍 {u}", fontsize=10, color="blue")
        plt.text(0.6, len(user_inputs) - i, f"🤖 {b}", fontsize=10, color="darkgreen")
    plt.axis('off')
    plt.title("Conversation Flow")
    plt.show()

"""### Fallback Responses"""

import random

# ✅ Fallback responses
fallback_responses = [
    "I'm not sure about that, but I can help with something else!",
    "I couldn't quite understand. Could you ask differently?",
    "That's outside my knowledge, but I can try looking it up!"
]

def get_fallback_response():
    """Returns a fallback response when the chatbot is unsure."""
    return random.choice(fallback_responses)

"""### Dynamic Formatting When Asked"""

# ✅ Formatting Response
def format_response(response, format_type):
    """Formats chatbot responses dynamically."""
    sentences = response.split(". ")

    if "subpoints" in format_type:
        return "\n- " + "\n- ".join(sentences)

    elif "subheadings" in format_type:
        return "\n".join(f"## {s.strip()}" for s in sentences if s.strip())

    elif "flowchart" in format_type:
        return "\n".join(f"⬇ {s.strip()}" for s in sentences)

    return response  # Default response if no formatting is requested

# ✅ Detects formatting type
def detect_format_request(user_input):
    """Detects user request for formatting (subpoints, subheadings, etc.)."""
    if "subpoints" in user_input.lower():
        return "subpoints"
    elif "subheadings" in user_input.lower():
        return "subheadings"
    elif "flowchart" in user_input.lower():
        return "flowchart"
    return "default"




