import os
from typing import List

from chainlit.types import AskFileResponse
from langchain.document_loaders import PyPDFLoader


def write_pdf_files(files: List[AskFileResponse], tmp_folder: str) -> List[str]:
    paths = []
    for file in files:
        file_path = os.path.join(tmp_folder, file.path)
        paths.append(file_path)
        with open(file_path, "wb") as f:
            f.write(file.content)

    return paths


def load_pdf_files(paths: List[str], splitter) -> List[str]:
    docs = [PyPDFLoader(path).load() for path in paths]
    splitted_docs = [splitter.split_documents(doc) for doc in docs]
    for doc in splitted_docs:
        for i, chunk in enumerate(doc, start=1):
            chunk.metadata["chunk"] = i
    splitted_docs = [chunk for doc in splitted_docs for chunk in doc]
    return splitted_docs
