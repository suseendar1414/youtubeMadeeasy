from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings()
video_url = "https://www.youtube.com/watch?v=qxQIcDrre1E"

def create_vec_db(video_url: str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap = 100)
    docs = text_splitter.split_documents(transcript)
    db = FAISS.from_documents(docs, embeddings)
    return docs

def queryfunc(db,query,k):
    docs = db.similarity_search(query, k=4)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = OpenAI(model="text-davinci-oo3")

    prompt = PromptTemplate(
        input_variables = ["question", "docs"]
        template = """you are a helpful Youtube assistant that can answer question about videos
        based on the video transcript.

        Answer the following question: {question}
        by searching the following video transcript: {docs}

        only use factual informatio  from the transcript to answer the question.

        If you feel like you dont have enough information to answer the question,say "i don't know".

        your answers should be detailed. 
        """
    )
    chain = LLMChain(llm=llm,prompt = prompt)

    response = chain.run(question = query,doc = docs_page_content)
    response = response.replace("/n","")
    return response

