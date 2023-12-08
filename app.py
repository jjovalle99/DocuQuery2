import os
from datetime import datetime

import chainlit as cl
from dotenv import load_dotenv
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone

from src.chain import setup_chain
from src.embeddings import setup_embeddings
from src.loaders import load_pdf_files, write_pdf_files
from src.vectorstore import setup_pinecone

load_dotenv()
set_llm_cache(InMemoryCache())  # Use in-memory cache for LLM
os.environ["LANGCHAIN_WANDB_TRACING"] = "true"  # Visbility for W&B

index = setup_pinecone()
embeddings = setup_embeddings()
vectorstore = Pinecone(index=index, embedding=embeddings, text_key="text")
prompt_chain = setup_chain()
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=100, length_function=len
)


@cl.on_chat_start
async def start_chat():
    files = await cl.AskFileMessage(
        content="Please upload a PDF file to answer questions from.",
        accept=["application/pdf"],
        max_size_mb=100,
        max_files=10,
        timeout=60 * 60 * 24 * 7 * 365,
    ).send()

    out = cl.Message(content=f"Loading {len(files)} file(s)...")
    await out.send()

    tmp_folder = f"/tmp/pdf_files_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    os.makedirs(tmp_folder, exist_ok=True)
    paths = write_pdf_files(files=files, tmp_folder=tmp_folder)
    splitted_docs = load_pdf_files(paths=paths, splitter=splitter)
    retriever = vectorstore.as_retriever(
        search_kwargs={"filter": {"source": {"$in": paths}}}
    )
    await retriever.aadd_documents(splitted_docs)

    cl.user_session.set("retriever", retriever)
    out.content = "File(s) loaded!"
    await out.update()


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
