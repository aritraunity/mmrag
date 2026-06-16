from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import tool

class ResponseGen:


    def __init__(self) -> None:
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a fashion stylist expert."),
                MessagesPlaceholder("history"),
                ("human", "{context}\n\nUser query: {query}")
            ]
        )
        self.parser = StrOutputParser()
        self.llm = ChatOllama(model='llama3.2:latest')
        self.history = []
        self.chain = self.prompt | self.llm | self.parser


    def get_context (self, mmrag_output):
        caption = ""
        for text in mmrag_output['metadatas'][0]:
            caption += text['caption']+"\n"
        return caption

    def gen_response (self, query, vision_results):
        response = self.chain.invoke({
            "context":self.get_context(vision_results),
            "query": query,
            "history": self.history
        })
        self.history.append(HumanMessage(content=query))
        self.history.append(AIMessage(content=response))
        return response