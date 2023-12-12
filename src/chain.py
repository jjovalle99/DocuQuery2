from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema.messages import SystemMessage
from langchain.schema.output_parser import StrOutputParser


def setup_chain():
    system_message = SystemMessage(content="You are a helpful assistant.")
    human_template = """Based on the following context generate an answer for the question. If the answer is not available say I dont know.
    Context: {context}

    Question: {question}

    Answer:"""  # noqa: E501
    human_message = HumanMessagePromptTemplate.from_template(template=human_template)  # noqa: E501
    llm = ChatOpenAI(model="gpt-3.5-turbo-1106", streaming=True)
    prompt = ChatPromptTemplate.from_messages(messages=[system_message, human_message])  # noqa: E501
    parser = StrOutputParser()
    chain = prompt | llm | parser
    return chain
