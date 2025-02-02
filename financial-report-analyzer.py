import streamlit as st
import numpy as np
import re
from datetime import datetime
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA

# Streamlit UI setup
st.set_page_config(page_title="Financial Report Analyzer", layout="wide")
st.title("📊 Financial Report Analysis ")

# Sidebar settings
with st.sidebar:
    st.header("Analysis Configuration")
    model_choice = st.selectbox("Select LLM Model", ["deepseek-r1:1.5b", "llama3.2:1b"], help="Choose the language model for analysis")
    analysis_types = st.multiselect("Analysis Focus", ["Financial Metrics", "Risk Analysis", "Market Analysis", "Performance Indicators", "Cash Flow Analysis", "Balance Sheet Analysis", "Ratio Analysis"], default=["Financial Metrics"])

# File upload and analysis
uploaded_file = st.file_uploader("Upload Financial Report (PDF)", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())

    with st.spinner("Processing document..."):
        loader = PDFPlumberLoader("temp.pdf")
        docs = loader.load()
        full_text = " ".join([doc.page_content for doc in docs])
        
        # Document-based Q&A system
        text_splitter = SemanticChunker(HuggingFaceEmbeddings())
        documents = text_splitter.split_documents(docs)
        
        embedder = HuggingFaceEmbeddings()
        vector = FAISS.from_documents(documents, embedder)
        retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        
        llm = Ollama(model=model_choice)
        
        # Enhanced Q&A prompt template
        financial_prompt = """
        You are a financial analyst. Provide a structured, clean, and concise analysis with the following sections:
       ### **Analysis Structure:**
        1. **Key Financial Metrics** (Revenue, Profit, Expenses, Cash Flow)
        2. **Trends & Changes** (Year-over-Year & Quarter-over-Quarter)
        3. **Risks & Challenges** (Debt, Market Risks, Operational Issues)
        4. **Actionable Insights & Recommendations** (Steps to improve financial health)
        
        ### **Response Guidelines:**
        - **Avoid unnecessary repetition.**
        - **Ensure proper formatting of financial figures.**
        - **Keep responses within 200 words.**
              
        ### **Context:**
        {context}

        ### **Question:**
        {question}

        ### **Generate a structured financial analysis:**
        """

        
        QA_PROMPT = PromptTemplate.from_template(financial_prompt)
        
        llm_chain = LLMChain(llm=llm, prompt=QA_PROMPT)
        combine_documents_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="context")
        qa = RetrievalQA(combine_documents_chain=combine_documents_chain, retriever=retriever)
        
        
        # Interactive Q&A
        st.header("💡 Interactive Analysis")
        
        suggested_questions = {
            "Financial Metrics": ["What are the key revenue trends?", "How has profitability evolved?", "What are the main cost drivers?"],
            "Risk Analysis": ["What are the primary risk factors?", "How is the company managing market risks?", "What are the key regulatory risks?"],
            "Market Analysis": ["What is the company's market position?", "How does it compare to competitors?", "What are the market opportunities?"]
        }
        
        all_suggested_questions = [q for analysis_type in analysis_types for q in suggested_questions.get(analysis_type, [])]
        
        selected_question = st.selectbox("Suggested Questions:", [""] + all_suggested_questions)
        user_input = st.text_input("Or ask your own question:", value=selected_question)
        
        # Display selected model before analysis
        st.subheader(f"Using Model: {model_choice}")
        
        if user_input:
            with st.spinner("Analyzing..."):
                response = qa(user_input)["result"]

                # Remove <think> artifacts and extra spaces
                clean_response = re.sub(r"<think>|</think>", "", response).strip()

           

                # Ensure numbers are formatted correctly
                clean_response = re.sub(r'(\d)([A-Za-z])', r'\1 \2', clean_response)  # Space between numbers and text
                clean_response = re.sub(r'(\d{1,3})(\d{3})', r'\1,\2', clean_response)  # Add commas to large numbers

                st.subheader("📊 Analysis Results")
                st.markdown(f"- {clean_response}")
                
                if st.button("Export Analysis"):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    export_data = {"question": user_input, "analysis": response}
                    st.download_button(label="Download Analysis", data=str(export_data), file_name=f"financial_analysis_{timestamp}.txt", mime="text/plain")

else:
    st.info("Please upload a financial report (PDF) to begin analysis.")