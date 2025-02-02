# 🚀 Guide to Local Financial Analysis with DeepSeek R1, Llama & Ollama (RAG-Based)

## 📘 Overview  
This project sets up a **local Retrieval-Augmented Generation (RAG) system** for **financial report analysis** using **DeepSeek R1, Llama, LangChain, FAISS, and Ollama** with **Streamlit** as the interface.  

The system allows users to **upload financial reports (PDFs)** and extract **key financial insights**, comparing responses between **DeepSeek R1 and Llama** in a **fully offline environment**.

---

## 🛠️ Installation  

### 1️⃣ **Clone the Repository**  
```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install -r requirements.txt

ollama pull deepseek-r1:1.5b  
ollama pull llama3:2b  

Run the code

streamlit run app.py
