import os
import json
import numpy as np
import faiss
import uvicorn
import google.generativeai as genai

from fastapi import FastAPI, HTTPException, Request
from linebot.v3 import WebhookHandler
from linebot.v3.messaging import (Configuration,
                                  ApiClient,
                                  MessagingApi,
                                  ReplyMessageRequest,
                                  TextMessage)
from linebot.v3.webhooks import (MessageEvent,
                                 TextMessageContent)
from linebot.v3.exceptions import InvalidSignatureError
from typing import List
from sentence_transformers import SentenceTransformer
from contextlib import asynccontextmanager

ACCESS_TOKEN = "O0Vi8xE7Wh3A6BahSUC6O0VKR7RxR0p27jHBl1h39OdH9/d3cEtmrS4QT91BUEDmmrRqLrUiKLVxlJcggXWQ/MwNBJttPBjKEw8Oifg9O06on+Ab3UzbvQ7E8W56z5GeOIHvROzUsRVagavLPiTIbwdB04t89/1O/w1cDnyilFU="
CHANNEL_SECRET = "701c99f9fa1b4d0261e6f4dedcce76c8"
GEMINI_API_KEY = "AIzaSyBrn9N8g0RnZrYhW-vFe3Tb2ytibKcsU3E"

configuration = Configuration(access_token=ACCESS_TOKEN)
handler = WebhookHandler(channel_secret=CHANNEL_SECRET)

class GeminiRAGSystem:
    def __init__(self, json_db_path: str):
        # Configure Gemini API
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Initialize Gemini model
        self.generation_model = genai.GenerativeModel('gemini-1.5-flash')
        
        # JSON Database path
        self.json_db_path = json_db_path
        
        # Embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load or create database
        self.load_database()
        
        # Create FAISS index
        self.create_faiss_index()
    
    def load_database(self):
        """Load existing database or create new"""
        try:
            with open(self.json_db_path, 'r') as f:
                self.database = json.load(f)
        except FileNotFoundError:
            self.database = {
                'documents': [],
                'embeddings': []
            }

    
    def save_database(self):
        """Save database to JSON file"""
        with open(self.json_db_path, 'w') as f:
            json.dump(self.database, f, indent=2)
     
    def add_document(self, text: str):
        """Add document to database with embedding"""
        # Generate embedding
        embedding = self.embedding_model.encode([text])[0]
        
        # Add to database
        self.database['documents'].append(text)
        self.database['embeddings'].append(embedding.tolist())
        
        # Save and update index
        self.save_database()
        self.create_faiss_index()
    
    def create_faiss_index(self):
        """Create FAISS index for similarity search"""
        if not self.database['embeddings']:
            return
        
        embeddings = np.array(self.database['embeddings'], dtype='float32')
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
    
    def retrieve_documents(self, query: str, top_k: int = 3):
        """Retrieve most relevant documents"""
        if not self.database['embeddings']:
            return []
        
        # Embed query
        query_embedding = self.embedding_model.encode([query])
        
        # Perform similarity search
        D, I = self.index.search(query_embedding, top_k)
        
        return [self.database['documents'][i] for i in I[0]]
    
    def generate_response(self, query: str):
        """Generate response using Gemini and retrieved documents"""
        # Retrieve relevant documents
        retrieved_docs = self.retrieve_documents(query)
        
        # Prepare context
        context = "\n\n".join(retrieved_docs)
        
        # Construct prompt
        full_prompt = f"""You are an AI assistant. 
        Use the following context to answer the question precisely:

        Context:
        {context}

        Question: {query}
        
        Provide a detailed and informative response based on the context."""
        
        # Generate response using Gemini
        try:
            response = self.generation_model.generate_content(full_prompt)
            return response.text, full_prompt
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def clear_database(self):
        """Clear database and save to JSON file"""
        self.database = {
            'documents': [],
            'embeddings': []
        }
        self.save_database()

gemini = GeminiRAGSystem("gemini_rag_database.json")

@asynccontextmanager
async def lifespan():

    # Sample documents to add to knowledge base
    sample_documents = [
        "จิรภัทร ทำดี คือ ชายหนุ่มที่มีความสามารถในการเขียนโปรแกรมที่มาจากบริษัท ClickNext ที่เป็นบริษัทด้านการพัฒนาโปรแกรมที่มีชื่อเสียง",
        "จิรภัทร ทำดี มีชื่อเล่นว่า ปาล์ม เกืดวันที่ 25 มกราคม 2545 ที่จังหวัดสระบุรี ศึกษาจบจากมหาวิทยาลัยธรรมศาสตร์ สาขาวิศวกรรมคอมพิวเตอร์",
        "งาน BUU-Bootcamp-2024 จัดที่มหาวิทยาลัยบูรพา ในวันที่ 25 มกราคม 2565 โดยมีการจัดกิจกรรมต่าง ๆ ที่เกี่ยวข้องกับการพัฒนาซอฟต์แวร์ เวลา 9:00 น. - 16:00 น.",
        "มหาวิทยาลัยบูรพา สาขาวิชาAI ปีการศึกษา 2565 มีนักศึกษาจำนวน 100 คน มีอาจารย์ที่ปรึกษา 10 คน"
    ]
    
    # Add documents to RAG system
    for doc in sample_documents:
        gemini.add_document(doc)

    yield
    gemini.clear_database()
    
app = FastAPI(lifespan=lifespan)

@app.get('/')
async def geeting():
    return "Hello from Backend 🙋‍♂️"

@app.get('/test-message')
async def test_message_gemini(request: Request,
                              text: str):
    """
    Debug message from gemini
    """
    response, prompt = gemini.generate_response(text)

    response_dict = {
        "gemini_answer": response,
        "full_prompt": prompt
    }

    return response_dict

@app.post('/message')
async def message(request: Request):
    signature = request.headers.get('X-Line-Signature')
    if not signature:
        raise HTTPException(
            status_code=400, detail="X-Line-Signature header is missing")

    body = await request.body()

    try:
        handler.handle(body.decode("UTF-8"), signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")


@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event: MessageEvent):
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)

        gemini_response, prompt = gemini.generate_response(event.message.text)

        line_bot_api.reply_message_with_http_info(
            ReplyMessageRequest(
                replyToken=event.reply_token,
                messages=[TextMessage(text=gemini_response)]
            )
        )

if __name__ == "__main__":

    uvicorn.run("main:app",
                port=8000,
                host="0.0.0.0")