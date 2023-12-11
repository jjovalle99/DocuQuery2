import logging
import os
from time import perf_counter

import chainlit as cl
from dotenv import load_dotenv
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone

from src.chain import setup_chain
from src.embeddings import setup_embeddings
from src.loaders import get_docs
from src.vectorstore import setup_pinecone

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    load_dotenv()
    set_llm_cache(InMemoryCache())  # Use in-memory cache for LLM
    os.environ["LANGCHAIN_WANDB_TRACING"] = "true"  # Visibility for W&B

    index = setup_pinecone()
    embeddings = setup_embeddings()
    vectorstore = Pinecone(index=index, embedding=embeddings, text_key="text")
    prompt_chain = setup_chain()
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
