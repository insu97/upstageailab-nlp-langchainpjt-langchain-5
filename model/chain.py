from langchain_upstage import ChatUpstage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def create_models(retriever, prompt, openai_api_key):
    """
    검색기와 프롬프트를 기반으로 LLM 체인을 생성합니다.
    """
    llm = ChatUpstage(api_key=openai_api_key, model="solar-pro", temperature=0)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
