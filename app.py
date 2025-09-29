# streamlit run streamlit_app.py
import os
os.environ["STREAMLIT_WATCH_DISABLE"] = "true"


import streamlit as st
import warnings
from langchain_core._api import LangChainDeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

import requests
import re
from typing import List, TypedDict
from langchain.schema import Document, BaseMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import Ollama
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.chains import create_history_aware_retriever
from langgraph.graph import StateGraph
from langchain_community.document_loaders import NotebookLoader


# Streamlit token streaming callback
class StreamlitDisplayCallback(BaseCallbackHandler):
    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.tokens = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.tokens += token
        self.placeholder.markdown(self.tokens)


# Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2")

@st.cache_data
def load_documents():
    owner = "Pulkit12dhingra"
    repo = "Blog"
    path = "content"
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    response = requests.get(api_url)
    response.raise_for_status()
    files = response.json()

    documents = []
    for file in files:
        if file["name"].endswith(".html"):
            raw_url = file["download_url"]
            html_content = requests.get(raw_url).text
            doc = Document(page_content=html_content, metadata={"source": file["name"]})
            documents.append(doc)
    return documents

# Load blog documents
documents = load_documents()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)
vectorstore = FAISS.from_documents(chunks, embedding=embedding_model)
retriever = vectorstore.as_retriever()

# Prompt to rephrase question based on chat history
retriever_prompt = (
    "Given a chat history and the latest user question which might reference context in the chat history, "
    "formulate a standalone question which can be understood without the chat history."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", retriever_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])
history_aware_retriever = create_history_aware_retriever(
    Ollama(model="qwen2.5-coder:7b"), retriever, contextualize_q_prompt
)

# LangGraph State
class GraphState(TypedDict):
    query: str
    chat_history: List[BaseMessage]
    documents: List
    reasoning: str
    answer: str

# LangGraph Nodes
def input_node(state: GraphState) -> GraphState:
    return state

def retrieve_node(state: GraphState) -> GraphState:
    question = state['query']
    history = state['chat_history']
    standalone_question = history_aware_retriever.invoke({
        "chat_history": history,
        "input": question
    })
    state["documents"] = standalone_question
    return state

def self_reasoning_node(state: GraphState) -> GraphState:
    reasoning_prompt = PromptTemplate.from_template(
        "Given the question: {query}\nand retrieved docs: {docs}\nWhat is a step-by-step reasoning path?"
    )
    reasoning_chain = reasoning_prompt | Ollama(model="qwen2.5-coder:7b")
    reasoning = reasoning_chain.invoke({"query": state["query"], "docs": state["documents"]})
    state["reasoning"] = reasoning
    return state

def generate_answer_node(state: GraphState) -> GraphState:
    full_context = f"{state['reasoning']}\n\n{state['documents']}"
    final_prompt = PromptTemplate.from_template(
        "Answer the question: {query}\nUse context:\n{context}"
    )
    placeholder = st.session_state["llm_placeholder"]
    callback = StreamlitDisplayCallback(placeholder)
    llm = Ollama(model="qwen2.5-coder:7b", callbacks=[callback])
    answer_chain = final_prompt | llm
    answer = answer_chain.invoke({"query": state["query"], "context": full_context})
    state["answer"] = answer
    return state

def output_node(state: GraphState) -> GraphState:
    return state

# Build LangGraph
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
app = graph.compile()

# UI
st.title("Jupyter Notebook → Blog Generator")

uploaded_notebook = st.file_uploader("Upload a Jupyter Notebook", type=["ipynb"])
if uploaded_notebook:
    with open("uploaded_notebook.ipynb", "wb") as f:
        f.write(uploaded_notebook.getvalue())

    notebook_loader = NotebookLoader("uploaded_notebook.ipynb", include_outputs=True, max_output_length=20, remove_newline=True)
    notebook_docs = notebook_loader.load()
    notebook_text = notebook_docs[0].page_content

    example_head = """<head>
      <meta charset="UTF-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      <title>Byte-Sized-Brilliance-AI</title>
      <link rel="icon" type="image/png" href="../img/icon.png" />
      <meta property="og:title" content="Pulkit's Blog" />
      <meta property="og:description" content=<place holder for description> />
      <meta property="og:image" content="../img/social-preview.jpg" />
      <meta property="og:type" content="website" />
      <meta name="twitter:card" content="summary_large_image" />
      <meta name="twitter:title" content="Pulkit's Blog" />
      <meta name="twitter:description" content=<place holder for description>/>
      <meta name="twitter:image" content="../img/social-preview.jpg" />
      <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet" />
      <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css" rel="stylesheet" />
      <link rel="stylesheet" href="data/style.css?v=1.0.0" />
      <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css" rel="stylesheet"/>
    </head>"""

    example_query = (
        "Based on the following Jupyter notebook python code, generate a sample HTML blog that explains all the code in the notebook content. "
        "Make sure to add all the code references that you are explaining along with the text\n"
        "In the head section of the blog update the <place holder for description> section with a short description of the blog\n"
        "Don't provide any other explanation or detail in the reply, only provide the html code starting from the head section\n"
        "Make sure the <head> section of the blog is exactly the same as this one:\n"
        f"{example_head}\n\n"
        "Notebook content:\n"
        f"{notebook_text}"
    )

    if "generating" not in st.session_state:
        st.session_state["generating"] = False

    clicked = st.button("Generate Blog", disabled=st.session_state["generating"])

    if clicked:
        st.session_state["generating"] = True
        st.session_state["llm_placeholder"] = st.empty()

        with st.spinner("Generating your blog... Please wait..."):
            result = app.invoke({
                "query": example_query,
                "chat_history": []
            })

        html_output = result["answer"]
        match = re.search(r"<head>.*?</body>", html_output, re.DOTALL | re.IGNORECASE)
        st.session_state["generating"] = False

        if match:
            cleaned_html = match.group(0)
            st.session_state["llm_placeholder"].empty()
            st.markdown("### Edit Blog Content Below")
            edited_blog = st.text_area("Blog HTML", value=cleaned_html, height=500)
            st.download_button("Download Edited Blog", edited_blog, file_name="blog.html", mime="text/html")
        else:
            st.error("Could not find a valid <head> to </body> block.")
