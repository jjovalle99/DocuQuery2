from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.storage import LocalFileStore


def setup_embeddings():
    store = LocalFileStore("./cache/")
    embeddings = OpenAIEmbeddings()
    return CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings=embeddings,
        document_embedding_cache=store,
        namespace=embeddings.model,
    )
