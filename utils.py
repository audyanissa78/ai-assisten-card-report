import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def process_document(uploaded_file):
  """
  menerima file PDF (rubrik/indikator), memecahnya, dan menyimpannya ke memori AI.
  """
  #1. Simpan file sementara
  temp_file = "tempt_rubrik.pdf"
  with open(temp_file, "wb") as f:
    f.write(uploaded_file.getvalue())

  #2. Load PDF
  loader = PyPDFLoader(temp_file)
  docs = loader.load()

  #3. Split Text (Pecah agar muat di memori)
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=220)
  splits = text_splitter.split_documents(docs)

  #4. Embeddings (Mengubah teks rubrik jadi angka)
  embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

  #5. Simpan ke Vektor Store (FAISS)
  vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)

  #hapus file sementara
  if os.path.exists(temp_file):
        os.remove(temp_file)

  return vectorstore
