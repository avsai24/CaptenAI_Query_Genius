from read_write import read_file
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os
from web_scraping import scrape_and_write_to_file , prepare_documents
import logging

logging.basicConfig(
    level=logging.INFO, 
    format="---------- %(levelname)s - %(message)s ----------", 
)
def embed_documents_with_chroma(documents, persist_directory):
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(documents)
    logging.info(f"completed text-splitter with length of {len(split_docs)}")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    logging.info("embeddings created")  
    Chroma.from_documents(split_docs, embeddings, persist_directory=persist_directory)
    logging.info("embeddings stored in vectordb")

def main(links_file_path,base_dir,chroma_dir):
    logging.info("started reading yaml file")
    yaml_data = read_file(links_file_path)
    logging.info("finished reading yaml file")
    links = yaml_data.get("links", [])
    logging.info(f"links found: {len(links)} and links are {links}")
    scrape_and_write_to_file(links,base_dir)
    logging.info("finished web scraping")
    documents = prepare_documents(base_dir)
    logging.info("doccuments created")
    os.makedirs(chroma_dir, exist_ok=True)
    embed_documents_with_chroma(documents, chroma_dir)

if __name__ == "__main__":
    links_file_path = "/Users/venkatasaiancha/Documents/all_concepts/multi_databse_retriver/vector_db_files/links.yaml"
    base_dir = "/Users/venkatasaiancha/Documents/all_concepts/multi_databse_retriver/vector_db_files/scraped_data" 
    chroma_dir = "/Users/venkatasaiancha/Documents/all_concepts/multi_databse_retriver/vector_db_files/chroma_db"
    main(links_file_path,base_dir,chroma_dir)