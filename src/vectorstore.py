import os

import pinecone


def setup_pinecone():
    pinecone.init(
        api_key=os.environ["PINECONE_API_KEY"], environment=os.environ["PINECONE_ENV"]
    )
    index = pinecone.Index(index_name=os.environ["PINECONE_INDEX"])
    return index
