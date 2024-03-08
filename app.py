from dotenv import load_dotenv

load_dotenv()

import logging
import os
from time import perf_counter

import chainlit as cl
from langchain.cache import InMemoryCache
from langchain.embeddings import CacheBackedEmbeddings
from langchain.globals import set_llm_cache
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Index

from src.loaders import get_docs

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ["LANGCHAIN_WANDB_TRACING"] = "true"

try:
    set_llm_cache(InMemoryCache())  # Use in-memory cache for LLM

    index = Index(
        api_key=os.environ["PINECONE_API_KEY"],
        index_name=os.environ["PINECONE_INDEX"],
        host=os.environ["PINECONE_HOST"],
    )

    store = LocalFileStore("./cache/")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    embeddings = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings=embeddings,
        document_embedding_cache=store,
        namespace=embeddings.model,
    )

    vectorstore = PineconeVectorStore(
        index=index, embedding=embeddings, text_key="text"
    )

    system_message = SystemMessage(content="You are a helpful assistant.")
    human_template = """Based on the following context generate an answer for the question. If the answer is not available say I dont know.
    Context: {context}

    Question: {question}

    Answer:"""
    human_message = HumanMessagePromptTemplate.from_template(template=human_template)
    llm = ChatOpenAI(model="gpt-3.5-turbo-1106", streaming=True)
    prompt = ChatPromptTemplate.from_messages(messages=[system_message, human_message])
    parser = StrOutputParser()
    prompt_chain = prompt | llm | parser
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100, length_function=len
    )
    logger.info("Initialization completed successfully.")
except Exception as e:
    logger.exception("Failed during initialization: %s", str(e))


@cl.on_chat_start
async def start_chat():
    files = await cl.AskFileMessage(
        content="Please upload a PDF file to answer questions from.",
        accept=["application/pdf"],
        max_size_mb=100,
        max_files=10,
        timeout=60 * 60 * 24 * 7 * 365,
    ).send()

    out = cl.Message(content="")
    await out.send()

    paths = [file.path for file in files]
    logger.info(files[0])
    logger.info("Preparing docs: %s", paths)
    start = perf_counter()
    splitted_docs = get_docs(files=files, splitter=splitter)
    end = perf_counter()
    logger.info("Preparing docs took %s seconds.", end - start)

    retriever = vectorstore.as_retriever(
        search_kwargs={"filter": {"source": {"$in": paths}}}
    )

    logger.info("Adding documents to vector store retriever.")
    start = perf_counter()
    await retriever.aadd_documents(splitted_docs)
    end = perf_counter()
    logger.info("Adding documents took %s seconds.", end - start)

    cl.user_session.set("retriever", retriever)
    out.content = f"{len(files)} file(s) loaded! You can now ask questions"
    await out.update()
    logger.info("Files loaded and retriever updated.")


@cl.on_message
async def main(message: cl.Message):
    retriever = cl.user_session.get("retriever")
    retriever_chain = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    )

    out = cl.Message(content="")
    await out.send()

    chain = retriever_chain | prompt_chain
    stream = chain.astream(message.content)

    async for chunk in stream:
        await out.stream_token(chunk)

    await out.update()
