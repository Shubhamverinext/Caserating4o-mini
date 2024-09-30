from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
import pandas as pd

DATA_PATH = 'data/annon_ratings.csv'
DB_FAISS_PATH = 'vectorstores/'

#embedding_model = 'sentence-transformers/all-mpnet-base-v2'
embedding_model = 'sentence-transformers/all-MiniLM-L6-v2'

# Create vector database
def create_vector_db(data_path, db_path):
    loader = DirectoryLoader(data_path,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                    chunk_overlap=0)
    # text_splitter = CharacterTextSplitter(chunk_size=2000,
    #                                                 chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    print(len(texts))
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model,
                                       model_kwargs={'device': 'cpu'})

    db = FAISS.from_documents(texts, embeddings)
    db.save_local(db_path)

def create_vector_db_xls(data_path, db_path):
    hr_df = pd.read_csv(data_path)
    print(hr_df.head())
    loader = DataFrameLoader(hr_df, page_content_column="Referral Name")
    documents = loader.load()
    print(len(documents))
 
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model,
                                       model_kwargs={'device': 'cpu'})

    db = FAISS.from_documents(documents, embeddings)
    db.save_local(db_path)
    
    
if __name__ == "__main__":
    create_vector_db_xls(DATA_PATH, DB_FAISS_PATH)
