import os
from settings import Settings
from text_processor import TextProcessor
from embedder import Embedder
from db_manager import DBManager
from query_handler import QueryHandler
from responder import Responder
from timings import logger
import time

class MemoryContext:
    def __init__(self, max_history=5):
        self.history = []
        self.max_history = max_history

    def add(self, question, answer):
        self.history.append((question, answer))
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def get_context(self):
        return "\n\n".join([f"Q: {q}\nA: {a}" for q, a in self.history])

def setup():
    try:
        Settings.check()
        
        embedder = Embedder()
        db = DBManager(embedder)
        
        # Check for new text files and process them
        new_files = check_for_new_files()
        if new_files:
            process_new_files(new_files, db)
        else:
            logger.info("No new files to process.")
        
        query_handler = QueryHandler(db)
        responder = Responder()
        memory = MemoryContext()
        
        logger.info("Setup complete")
        return query_handler, responder, memory
    except Exception as e:
        logger.error(f"Setup error: {str(e)}")
        raise

def check_for_new_files():
    processed_files = set()
    processed_files_path = os.path.join('data', 'processed_files.txt')
    
    if os.path.exists(processed_files_path):
        with open(processed_files_path, 'r') as f:
            processed_files = set(f.read().splitlines())
    
    data_folder = os.path.join('data')
    current_files = set(f for f in os.listdir(data_folder) if f.endswith('.txt'))
    new_files = current_files - processed_files
    return new_files

def process_new_files(new_files, db):
    data_folder = os.path.join('data')
    
    for file in new_files:
        file_path = os.path.join(data_folder, file)
        logger.info(f"Processing new file: {file_path}")
        
        documents = TextProcessor.process_text(file_path)
        db.add_docs(documents)
        
        # Mark file as processed
        with open(os.path.join('data', 'processed_files.txt'), 'a') as f:
            f.write(f"{file}\n")


def run():
    try:
        query_handler, responder, memory = setup()
        
        while True:
            query = input("Ask a question (or 'quit' to exit): ")
            if query.lower() == 'quit':
                break
            
            context = query_handler.handle(query)
            memory_context = memory.get_context()
            
            full_context = f"{memory_context}\n\n{context}"
            
            answer = responder.respond(query, full_context)
            print(f"Answer: {answer}")
            
            if "technical difficulties" not in answer and "unable to provide an answer" not in answer:
                memory.add(query, answer)
            else:
                print("Apologies for the inconvenience. Please wait a moment before asking another question.")
                time.sleep(10)  
    except Exception as e:
        logger.error(f"Runtime error: {str(e)}")

if __name__ == "__main__":
    run()