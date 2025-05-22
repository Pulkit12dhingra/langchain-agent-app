# %%
# load the data
# first we'll load all the blogs using 

import requests
from langchain.document_loaders import BSHTMLLoader
from langchain.schema import Document

# GitHub repo details
owner = "Pulkit12dhingra"
repo = "Blog"
path = "content"
api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"

# List files in the content directory
response = requests.get(api_url)
response.raise_for_status()
files = response.json()

# Collect LangChain documents
documents = []

for file in files:
    if file["name"].endswith(".html"):
        print(f"Processing: {file['name']}")
        raw_url = file["download_url"]
        html_content = requests.get(raw_url).text

        # Load into LangChain using BSHTMLLoader
        loader = BSHTMLLoader(file_path=None)
        # Manually override the content for in-memory loading
        doc = Document(page_content=html_content, metadata={"source": file["name"]})
        documents.append(doc)

print(f"\nLoaded {len(documents)} HTML documents into LangChain.")


# %%
# split the document list into chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Chunking
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# %%
from langchain.embeddings import HuggingFaceEmbeddings

# Embedding
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# %%
from langchain.vectorstores import FAISS

vectorstore = FAISS.from_documents(chunks, embedding=embedding_model)
retriever = vectorstore.as_retriever()

# %%
from langchain.llms import Ollama

from langchain.callbacks.base import BaseCallbackHandler
# Custom callback handler to print word-by-word in real-time
class WordByWordPrinter(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(token, end='', flush=True)

# LLM initialization
llm = Ollama(model="qwen2.5-coder:7b",callbacks=[WordByWordPrinter()])


# %%
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever

# History-aware retriever
retriever_prompt = (
    "Given a chat history and the latest user question which might reference context in the chat history, "
    "formulate a standalone question which can be understood without the chat history. "
    "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", retriever_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# %%

from langchain.schema import BaseMessage
from typing import TypedDict, List

# Define the shared state for LangGraph
class GraphState(TypedDict):
    query: str
    chat_history: List[BaseMessage]
    documents: List
    reasoning: str
    answer: str

# %%
# Input node
def input_node(state: GraphState) -> GraphState:
    return state


# %%
# Retrieval node
def retrieve_node(state: GraphState) -> GraphState:
    question = state['query']
    history = state['chat_history']
    standalone_question = history_aware_retriever.invoke({"chat_history": history, "input": question})
    state["documents"] = standalone_question
    return state

# %%
from langchain_core.prompts import PromptTemplate
# Self-reasoning node
def self_reasoning_node(state: GraphState) -> GraphState:
    reasoning_prompt = PromptTemplate.from_template(
        "Given the question: {query}\nand retrieved docs: {docs}\nWhat is a step-by-step reasoning path?"
    )
    reasoning_chain = reasoning_prompt | llm
    reasoning = reasoning_chain.invoke({"query": state["query"], "docs": state["documents"]})
    state["reasoning"] = reasoning
    return state

# %%
# Answer generation node
def generate_answer_node(state: GraphState) -> GraphState:
    full_context = f"{state['reasoning']}\n\n{state['documents']}"
    final_prompt = PromptTemplate.from_template(
        "Answer the question: {query}\nUse context:\n{context}"
    )
    answer_chain = final_prompt | llm
    answer = answer_chain.invoke({"query": state["query"], "context": full_context})
    state["answer"] = answer
    return state

# %%

# Output node
def output_node(state: GraphState) -> GraphState:
    return state

# %%
from langgraph.graph import StateGraph

# Build the LangGraph
graph = StateGraph(GraphState)
graph.add_node("input", input_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("reason", self_reasoning_node)
graph.add_node("generate", generate_answer_node)
graph.add_node("output", output_node)

graph.set_entry_point("input")
graph.add_edge("input", "retrieve")
graph.add_edge("retrieve", "reason")
graph.add_edge("reason", "generate")
graph.add_edge("generate", "output")
graph.set_finish_point("output")

# %%
# Compile app
app = graph.compile()

# %%
# Example invocation
example_query = "Summarize the main points of the documents."
example_history = []  # Fill with actual history if available

result = app.invoke({
    "query": example_query,
    "chat_history": example_history
})

print("Final Answer:", result["answer"])

# %%


# %%
# %%
# Step 1: Load the notebook using LangChain's NotebookLoader
from langchain_community.document_loaders import NotebookLoader

# Load the Jupyter notebook file
notebook_loader = NotebookLoader(
    "sample_notebook.ipynb",     # <-- Replace with your notebook path
    include_outputs=True,        # Include cell outputs like print, plots, etc.
    max_output_length=20,        # Truncate output to avoid overflow
    remove_newline=True          # Strip extra newlines
)

notebook_docs = notebook_loader.load()  # Returns a list of Document objects
notebook_text = notebook_docs[0].page_content  # Use the content of the first (and usually only) doc

# %%
# Step 2: Extract <head> section from one of the previously loaded HTML documents
example_head = """<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Byte-Sized-Brilliance-AI</title>
  <link rel="icon" type="image/png" href="../img/icon.png" />

  <!-- Meta for sharing -->
  <meta property="og:title" content="Pulkit's Blog" />
  <meta property="og:description" content=<place holder for description> />
  <meta property="og:image" content="../img/social-preview.jpg" />
  <meta property="og:type" content="website" />
  <meta name="twitter:card" content="summary_large_image" />
  <meta name="twitter:title" content="Pulkit's Blog" />
  <meta name="twitter:description" content=<place holder for description>/>
  <meta name="twitter:image" content="../img/social-preview.jpg" />

  <!-- CSS and Bootstrap -->
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet" />
  <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css" rel="stylesheet" />
  <link rel="stylesheet" href="data/style.css?v=1.0.0" />
  <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css" rel="stylesheet"/>
</head>"""

# %%
# Step 3: Construct the query using the notebook content and HTML <head>
example_query = (
    "Based on the following Jupyter notebook python code, generate a sample HTML blog that explains all the code in the notebook content. Make sure to add all the code references that you are explaining along with the text\n"
    "In the head section of the blog update the <place holder for description> section with a short description of the blog\n"
    "Don't provide any other explaination or detail in the reply, only provide the html code starting from the head section\n"
    "Make sure the <head> section of the blog is exactly the same as this one:\n"
    f"{example_head}\n\n"
    "Notebook content:\n"
    f"{notebook_text}"
)

example_history = []

# %%
# Step 4: Run the LangGraph app with the new query
result = app.invoke({
    "query": example_query,
    "chat_history": example_history
})

# Output the generated HTML blog
print("\n\nFinal Answer (cached):\n", result["answer"])

# %%
# %%
import re

# Extract content from <head> to </body>
html_output = result["answer"]

# Use regex to find <head> ... </body> block (non-greedy match)
match = re.search(r"<head>.*?</body>", html_output, re.DOTALL | re.IGNORECASE)

if match:
    html_content = match.group(0)
    
    # Save to HTML file
    output_path = "generated_blog.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\n Extracted and saved blog to {output_path}")
else:
    print("\n Could not find complete <head> to </body> block in the response.")


# %%



