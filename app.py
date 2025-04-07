import os
import streamlit as st
from dotenv import load_dotenv

# 모듈 임포트
from function.pre_process import (
    load_documents,
    split_documents,
    embedding,
    sementic_chunker,
    vectorstore,
    vectorstore_sementic,
    retriever,
    create_models,
)

from langchain_core.prompts import PromptTemplate
from langchain_upstage import UpstageGroundednessCheck
from langchain_community.retrievers import TavilySearchAPIRetriever


if __name__ == "__main__":

    load_dotenv()
    st.session_state.openai_api_key = os.getenv("OPENAI_API_KEY")
    st.session_state.tavily_api_key = os.getenv("TAVILY_API_KEY")

    st.set_page_config(layout="wide")
    st.title("LangChain Project")

    col1, col2, col3 = st.columns([1, 1, 2], border=True)

    with col1:

        st.subheader("Settings")

        st.markdown("---")

        st.session_state.chunk_size = st.number_input("chunk_size", value=1000, step=50)
        st.session_state.chunk_overlap = st.number_input("chunk_overlap", value=250, step=50)

        # 라디오 버튼으로 답변 생성 방식을 선택합니다.
        st.session_state.method = st.radio("답변 생성 방식 선택", ("임베딩 기반", "시멘틱 청커 기반"))

        prompt_text = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Answer in Korean.

#Question: 
{question} 
#Context: 
{context} 

#Answer:"""
        
        st.session_state.new_prompt = st.text_area("프롬프트를 작성해주세요!", value=prompt_text, height=350, key="prompt_write")

        if "rag_state" not in st.session_state:
            st.session_state.rag_state = False

        if st.button("실행"):
            st.session_state.rag_state = True

    with col2:

        st.subheader("Result")

        st.markdown("---")

        if st.session_state.rag_state:
            st.session_state.pdf_files, st.session_state.all_docs = load_documents()
            st.session_state.split_documents = split_documents(all_docs=st.session_state.all_docs, chunk_size=st.session_state.chunk_size, chunk_overlap=st.session_state.chunk_overlap)
            st.session_state.embeddings  = embedding(openai_api_key=st.session_state.openai_api_key)

            # 3. 사용자가 선택한 방식에 따라 파이프라인 구성
            if st.session_state.method == "임베딩 기반":
                st.session_state.vectorstore = vectorstore(
                    st.session_state.embeddings, st.session_state.split_documents
                )
                st.session_state.retriever = retriever(st.session_state.vectorstore)
            else:  # 시멘틱 청커 기반
                # 시멘틱 청커는 비용이 많이 드는 작업일 수 있으므로, 결과를 캐싱해서 재사용합니다.
                if "semantic_chunks" not in st.session_state:
                    st.session_state.semantic_chunks = sementic_chunker(
                        st.session_state.embeddings, st.session_state.split_documents
                    )
                st.session_state.vector_store_instance = vectorstore_sementic(
                    st.session_state.embeddings, st.session_state.semantic_chunks
                )
                st.session_state.retriever = retriever(st.session_state.vector_store_instance)

            st.session_state.prompt = PromptTemplate.from_template(st.session_state.new_prompt)
            st.session_state.chain = create_models(st.session_state.retriever, st.session_state.prompt, st.session_state.openai_api_key)

            st.write(f"➡️ 총 {len(st.session_state.pdf_files)}개의 PDF 파일에서 {len(st.session_state.all_docs)}개의 문서를 로드했습니다.")
            st.write(f"➡️ 분할된 청크의 수: {len(st.session_state.split_documents)}")
            if st.session_state.method == "시멘틱 청커 기반":
                st.write(f"➡️ 시멘틱 청크의 수: {len(st.session_state.semantic_chunks)}")
            st.text_area("➡️ 프롬프트 내용", f"{st.session_state.prompt.template}", height=350)

    with col3:

        st.subheader("QA")

        st.markdown("---")

        st.session_state.question = st.text_area("질문을 해주세요!", "농구에서 3점 슛 기준에 대해서 bullet points 형식으로 작성해 주세요.", height=200)
        if st.button("실행", key="execute"):
                response = st.session_state.chain.invoke(st.session_state.question)

                st.markdown("---")

                st.subheader("Groundedness Check")

                # 분할된 문서 내용도 하나의 긴 문자열로 결합합니다.
                relevant_docs = st.session_state.retriever.get_relevant_documents(st.session_state.question)
                pdf_contents = "\n\n".join([doc.page_content for doc in relevant_docs])

                # request_input에 context와 생성된 답변(response)를 할당합니다.
                request_input = {
                    "context": pdf_contents,
                    "answer": response,
                }

                st.session_state.groundedness_check = UpstageGroundednessCheck(
                    api_key=st.session_state.openai_api_key,
                    model="solar-pro",
                    temperature=0,
                    base_url="https://api.upstage.ai/v1",
                )

                # Groundedness Check 실행
                st.session_state.gc_result  = st.session_state.groundedness_check.invoke(request_input)

                if st.session_state.gc_result.lower().startswith("grounded"):
                    st.text("✅ Groundedness check passed")
                    st.markdown(response)
                else:
                    st.text("❌ Groundedness check failed")
                    st.text("🔍 Tavily API를 사용하여 웹 검색 결과를 가져옵니다.")
                    st.session_state.tavily_retriever = TavilySearchAPIRetriever(k=3)
                    tavily_chain = create_models(
                        st.session_state.tavily_retriever,
                        st.session_state.prompt,
                        st.session_state.openai_api_key
                    )
                    # tavily_chain을 통해 질문에 대한 답변을 생성합니다.
                    tavily_response = tavily_chain.invoke(st.session_state.question)
                    st.text(tavily_response)