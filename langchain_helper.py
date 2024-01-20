from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate 
from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_openai import OpenAI
import os


# Load environment variables
load_dotenv()

# # Assuming you have set your OpenAI API key in the .env file or as an environment variable
# openai.api_key = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings()

def create_vec_db(video_url: str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
    docs = text_splitter.split_documents(transcript)
    db = FAISS.from_documents(docs, embeddings)
    return db


def queryfunc(db, query, k):
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = OpenAI()

    prompt = PromptTemplate(
        input_variables=["question", "docs_page_content"],
        template="""You are a helpful Youtube assistant that can answer questions about videos
based on the video transcript.

Answer the following question: {question}
by searching the following video transcript: {docs_page_content}

Only use factual information from the transcript to answer the question.

If you feel like you don't have enough information to answer the question, say 'I don't know'.

Your answers should be detailed."""
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    # Note the change here: 'docs' is replaced with 'docs_page_content'
    response = chain.run(question=query, docs_page_content=docs_page_content)
    response = response.replace("\n", "")
    return response



