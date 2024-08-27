import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain import hub
from langchain.agents import create_openai_tools_agent, AgentExecutor
import requests

# Load environment variables
load_dotenv()

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = "paste yor api key here"
os.environ["LANGCHAIN_TRACING"] = "true"


# Set up tools
@st.cache_resource
def setup_tools(webpage_url):
    # Wikipedia tool
    wiki_api_wrapper = WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=300)
    wiki = WikipediaQueryRun(api_wrapper=wiki_api_wrapper)

    # Arxiv tool
    arxiv_wrapper = ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=300)
    arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

    # Web page retriever tool
    try:
        loader = WebBaseLoader(webpage_url)
        docs = loader.load()
        documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
        vectordb = FAISS.from_documents(documents, OpenAIEmbeddings())
        retriever = vectordb.as_retriever()
        retriever_tool = create_retriever_tool(
            retriever,
            "webpage_search",
            f"Search for information from the webpage: {webpage_url}. For any questions about this webpage, you must use this tool!"
        )
        return [wiki, arxiv, retriever_tool]
    except requests.exceptions.RequestException as e:
        st.error(f"Error loading webpage: {e}")
        return [wiki, arxiv]


# Streamlit UI
st.title("Advanced LLM Assistant")

st.sidebar.header("About")
st.sidebar.info(
    "This app uses LangChain to provide information from Wikipedia, ArXiv, "
    "and a user-defined webpage. It can answer questions, summarize papers, "
    "and provide detailed information on various topics."
)

# User input for webpage URL
webpage_url = st.text_input("Enter a webpage URL to analyze (optional):", "https://docs.smith.langchain.com/")

# Set up tools with user-defined webpage
tools = setup_tools(webpage_url)

# Set up LLM
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

# Set up agent
prompt = hub.pull("hwchase17/openai-functions-agent")
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

query = st.text_input("Enter your question:")

if query:
    with st.spinner("Thinking..."):
        try:
            response = agent_executor.invoke({"input": query})

            st.subheader("Response:")
            st.write(response["output"])

            st.subheader("Thought Process:")
            thought_process = response.get("intermediate_steps", [])
            for i, thought in enumerate(thought_process, 1):
                st.markdown(f"**Step {i}:**")
                st.write(f"Action: {thought[0]}")
                st.write(f"Observation: {thought[1]}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Additional features
st.header("Additional Features")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Paper Summarizer")
    paper_id = st.text_input("Enter ArXiv paper ID (e.g., 1605.08386):")
    if paper_id:
        with st.spinner("Summarizing paper..."):
            try:
                summary = agent_executor.invoke(
                    {"input": f"Summarize the key points of the paper {paper_id} in bullet points"})
                st.write(summary["output"])
            except Exception as e:
                st.error(f"An error occurred while summarizing the paper: {e}")

with col2:
    st.subheader("Webpage Info")
    webpage_query = st.text_input("Ask about the webpage you provided:")
    if webpage_query:
        with st.spinner("Searching webpage..."):
            try:
                webpage_info = agent_executor.invoke(
                    {"input": f"Tell me about {webpage_query} from the provided webpage"})
                st.write(webpage_info["output"])
            except Exception as e:
                st.error(f"An error occurred while searching the webpage: {e}")

st.header("Comparison Tool")
topic1 = st.text_input("Enter first topic:")
topic2 = st.text_input("Enter second topic:")
if topic1 and topic2:
    with st.spinner("Comparing topics..."):
        try:
            comparison = agent_executor.invoke({"input": f"Compare and contrast {topic1} and {topic2}"})
            st.write(comparison["output"])
        except Exception as e:
            st.error(f"An error occurred while comparing topics: {e}")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit and LangChain")