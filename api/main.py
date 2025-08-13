import os
import json
import hashlib
import secrets
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import traceback
import sqlite3
from flask import request, session, jsonify
import sqlite3
from fastapi import FastAPI, Header, Request, Form, UploadFile, File as FastAPIFile, HTTPException, Depends, Cookie, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, EmailStr
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
import os
from openai import OpenAI
# from models import Base, Chat
from fastapi import File
from fpdf import FPDF
import io
from fastapi import File, UploadFile

# --- All necessary agno imports ---
from agno.agent import Agent
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.models.openai import OpenAIChat
from agno.memory.v2.db.sqlite import SqliteMemoryDb
from agno.memory.v2.memory import Memory
from agno.storage.sqlite import SqliteStorage
from agno.media import File
from fastapi import Body
# Load environment variables from .env file
load_dotenv()

import openai
try:
    stt_client = openai.AsyncOpenAI()
    chat_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    print("OpenAI clients initialized successfully.")
except Exception as e:
    print(f"Error initializing OpenAI clients: {e}")
    stt_client = None
    chat_client = None

# --- Pydantic Models for Authentication and Chat ---
class UserSignup(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6)
    preferred_language: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    email: str
    preferred_language: str

class ChatRequest(BaseModel):
    name: str
    messages: List[Dict]

class ChatUpdate(BaseModel):
    messages: List[Dict]

# --- Database and Memory Configuration ---
os.makedirs("tmp", exist_ok=True)

os.makedirs("templates", exist_ok=True)

DB_FILE = "tmp/health_agent.db"

USERS_DB_FILE = "tmp/users.db"

# Initialize memory database and memory objects
memory_db = SqliteMemoryDb(table_name="user_memories", db_file=DB_FILE)
memory = Memory(db=memory_db, model=OpenAIChat(id="gpt-4o"))
storage = SqliteStorage(table_name="agent_sessions", db_file=DB_FILE)

# Create the database tables if they don't exist
try:
    memory_db.create()
    storage.create()
except Exception as e:
    print(f"Database initialization warning: {e}")

# --- AI-powered Chat Name Generation ---
async def generate_chat_name_with_ai(first_user_message: str, user_language: str = "english") -> str:
    """Generate a meaningful chat name using AI based on the first user message."""
    if not chat_client or not first_user_message.strip():
        return "New Chat"
    
    try:
        # Create a prompt to generate a chat name
        prompt = f"""
        Based on the following user message, create a short, meaningful chat name (maximum 40 characters).
        The chat name should be in {user_language} and capture the main topic or intent of the message.
        
        User message: "{first_user_message.strip()}"
        
        Guidelines:
        - Keep it under 40 characters
        - Make it descriptive but concise
        - Use {user_language} language
        - Don't include quotes or special formatting
        - Focus on the main health topic or concern
        
        Just provide the chat name, nothing else.
        """
        
        response = chat_client.chat.completions.create(
            model="gpt-4o-mini",  # Using mini for cost efficiency
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert at creating concise, meaningful titles. Generate only the requested chat name without any additional text or formatting."
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=50,
            temperature=0.3,  # Lower temperature for more consistent results
        )
        
        generated_name = response.choices[0].message.content.strip()
        
        # Clean up the generated name
        generated_name = generated_name.strip('"').strip("'").strip()
        
        # Fallback if generated name is too long or empty
        if len(generated_name) > 40:
            generated_name = generated_name[:37] + "..."
        
        if not generated_name or len(generated_name) < 3:
            return create_fallback_name(first_user_message)
        
        return generated_name
        
    except Exception as e:
        print(f"Error generating chat name with AI: {e}")
        return create_fallback_name(first_user_message)

def create_fallback_name(first_user_message: str) -> str:
    """Create a fallback chat name when AI generation fails."""
    if not first_user_message.strip():
        return "New Chat"
    
    # Clean up the message content
    cleaned_content = first_user_message.strip()
    
    # Remove common prefixes
    prefixes_to_remove = [
        'hi ', 'hello ', 'hey ', 'hi,', 'hello,', 'hey,',
        'what ', 'how ', 'can you ', 'could you ', 'please ',
        'i need ', 'i want ', 'help me ', 'can you help ',
        'i have a question about ', 'i would like to know ',
        'tell me about ', 'explain ', 'what is ', 'what are '
    ]
    
    cleaned_lower = cleaned_content.lower()
    for prefix in prefixes_to_remove:
        if cleaned_lower.startswith(prefix):
            cleaned_content = cleaned_content[len(prefix):].strip()
            break
    
    # Capitalize first letter
    if cleaned_content:
        cleaned_content = cleaned_content[0].upper() + cleaned_content[1:] if len(cleaned_content) > 1 else cleaned_content.upper()
    
    # Limit length
    if len(cleaned_content) > 40:
        words = cleaned_content[:37].split(' ')
        if len(words) > 1:
            cleaned_content = ' '.join(words[:-1]) + '...'
        else:
            cleaned_content = cleaned_content[:37] + '...'
    
    return cleaned_content if cleaned_content else "New Chat"

# --- User and Chat Database Management ---
class UserDatabase:
    def __init__(self, db_file: str):
        self.db_file = db_file
        self.init_db()
    
    def init_db(self):
        """Initialize the users and chats database tables."""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                preferred_language TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_sessions (
                session_token TEXT PRIMARY KEY,
                user_email TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                FOREIGN KEY (user_email) REFERENCES users (email)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_email TEXT NOT NULL,
                name TEXT NOT NULL,
                messages TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_email) REFERENCES users (email)
            )
        ''')
        conn.commit()
        conn.close()

    def hash_password(self, password: str) -> str:
        """Hash a password using SHA-256 with salt."""
        salt = secrets.token_hex(32)
        return hashlib.sha256((password + salt).encode()).hexdigest() + ':' + salt

    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify a password against its hash with debug info."""
        try:
            print(f"=== PASSWORD VERIFICATION DEBUG ===")
            print(f"Input password length: {len(password)}")
            print(f"Stored hash format valid: {':' in password_hash}")

            hash_part, salt = password_hash.split(':')

            computed_hash = hashlib.sha256((password + salt).encode()).hexdigest()

            print(f"Hash matches: {computed_hash == hash_part}")

            return computed_hash == hash_part

        except Exception as e:
            print(f"❌ Password verification error: {e}")
            return False
        
    def create_user(self, email: str, password: str, preferred_language: str) -> bool:
        """Create a new user."""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        try:
            password_hash = self.hash_password(password)
            cursor.execute(
                'INSERT INTO users (email, password_hash, preferred_language) VALUES (?, ?, ?)',
                (email, password_hash, preferred_language)
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
        finally:
            conn.close()
    
    def authenticate_user(self, email: str, password: str) -> Optional[dict]:
        """Authenticate a user and return user info if successful."""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        try:
            cursor.execute('SELECT email, password_hash, preferred_language FROM users WHERE email = ?', (email,))
            user_data = cursor.fetchone()
            if user_data and self.verify_password(password, user_data[1]):
                cursor.execute('UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE email = ?', (email,))
                conn.commit()
                return {'email': user_data[0], 'preferred_language': user_data[2]}
            return None
        finally:
            conn.close()

    def create_session(self, user_email: str) -> str:
        """Create a new session token for a user."""
        session_token = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(days=30)
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        try:
            cursor.execute(
                'INSERT INTO user_sessions (session_token, user_email, expires_at) VALUES (?, ?, ?)',
                (session_token, user_email, expires_at)
            )
            conn.commit()
            return session_token
        finally:
            conn.close()

    def get_user_by_session(self, session_token: str) -> Optional[dict]:
        """Get user info by session token."""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        try:
            cursor.execute('''
                SELECT u.email, u.preferred_language 
                FROM users u 
                JOIN user_sessions s ON u.email = s.user_email 
                WHERE s.session_token = ? AND s.expires_at > CURRENT_TIMESTAMP
            ''', (session_token,))
            user_data = cursor.fetchone()
            if user_data:
                return {'email': user_data[0], 'preferred_language': user_data[1]}
            return None
        finally:
            conn.close()

    def delete_session(self, session_token: str):
        """Delete a session (logout)."""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        try:
            cursor.execute('DELETE FROM user_sessions WHERE session_token = ?', (session_token,))
            conn.commit()
        finally:
            conn.close()
    
    def get_chats_for_user(self, user_email: str) -> List[Dict]:
        conn = sqlite3.connect(self.db_file)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        try:
            cursor.execute('SELECT id, name, created_at, updated_at FROM chats WHERE user_email = ? ORDER BY updated_at DESC', (user_email,))
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_chat_by_id(self, chat_id: int, user_email: str) -> Optional[Dict]:
        conn = sqlite3.connect(self.db_file)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        try:
            cursor.execute('SELECT * FROM chats WHERE id = ? AND user_email = ?', (chat_id, user_email,))
            row = cursor.fetchone()
            if row:
                chat_data = dict(row)
                chat_data['messages'] = json.loads(chat_data['messages'])
                return chat_data
            return None
        finally:
            conn.close()

    async def create_new_chat(self, user_email: str, messages: List[Dict], user_language: str = "english") -> int:
        """Create a new chat with AI-generated name."""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        try:
            # Find first user message for name generation
            first_user_message = ""
            for msg in messages:
                if (isinstance(msg, dict) and 
                    (msg.get('sender') in ['user', 'human'] or 
                     msg.get('role') == 'user')):
                    content = msg.get('content', '').strip()
                    if content and not content.lower().startswith('welcome'):
                        first_user_message = content
                        break
            
            # Generate chat name using AI
            name = await generate_chat_name_with_ai(first_user_message, user_language)
            
            messages_json = json.dumps(messages)
            cursor.execute(
                'INSERT INTO chats (user_email, name, messages) VALUES (?, ?, ?)', 
                (user_email, name, messages_json)
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def create_new_chat_with_name(self, user_email: str, name: str, messages: List[Dict]) -> int:
        """Create a new chat with a specific name."""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        try:
            messages_json = json.dumps(messages)
            cursor.execute(
                'INSERT INTO chats (user_email, name, messages) VALUES (?, ?, ?)', 
                (user_email, name, messages_json)
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def update_chat(self, chat_id: int, user_email: str, messages: List[Dict]) -> bool:
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        try:
            messages_json = json.dumps(messages)
            cursor.execute('UPDATE chats SET messages = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ? AND user_email = ?', (messages_json, chat_id, user_email))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

# Initialize user database
user_db = UserDatabase(DB_FILE)

# --- Authentication dependency ---
async def get_current_user(session_token: str = Cookie(None)) -> Optional[dict]:
    """Get current authenticated user from session token."""
    if not session_token:
        return None
    return user_db.get_user_by_session(session_token)


# --- MedicalKnowledgeBase ---
class MedicalKnowledgeBase:
    """Medical knowledge base with evidence-based guidelines."""
    def __init__(self):
        self.guidelines = {
            "emergency_symptoms": ["chest pain", "difficulty breathing", "severe headache", "stroke symptoms", "severe bleeding"],
            "diabetes_management": {"hba1c_normal": "<7.0%", "blood_sugar_normal": "80-130 mg/dL"},
            "hypertension": {"target_bp": "<140/90 mmHg", "high_risk": ">180/120 mmHg"},
            "general_advice": "Always consult healthcare professionals for medical decisions"
        }

# --- HealthAgent Class ---
class HealthAgent:
    """Orchestrator for AI health agents."""
    def __init__(self):
        self.knowledge_base = MedicalKnowledgeBase()
        self.language_patterns = {
            'hindi': ['क्या', 'कैसे', 'मुझे', 'आप', 'है', 'हैं', 'मेरा', 'डॉक्टर'],
            'gujarati': ['શું', 'કેમ', 'મને', 'તમે', 'છે', 'છો', 'ડૉક્ટર', 'સ્વાસ્થ્ય'],
            'english': ['what', 'how', 'can', 'you', 'is', 'are', 'doctor', 'health', 'medical'],
            'tamil': ['என்ன', 'எப்படி', 'நான்', 'நீங்கள்', 'இருக்கிறது', 'மருத்துவர்'],
            'telugu': ['ఏమి', 'ఎలా', 'నేను', 'మీరు', 'ఉంది', 'డాక్టర్']
        }
        self.agent_cache = {}
        self.conversation_history = {}

    def _create_general_agent(self, preferred_language: str, session_id: str) -> Agent:
        """Creates the main agent with persistent storage and memory."""
        if session_id in self.agent_cache:
            return self.agent_cache[session_id]
        
        language_instruction = f"""
        You are Satya, a compassionate AI health assistant. 
        IMPORTANT: Always respond in {preferred_language.title()} language unless the user specifically asks for another language.
        Your instructions are to remember the conversation history and provide helpful, accurate health information.
        Always emphasize that users should consult healthcare professionals for serious medical decisions.
        Be culturally sensitive and respectful in your responses.
        """
        
        agent = Agent(
            model=OpenAIChat(id="gpt-4o"),
            tools=[DuckDuckGoTools()],
            memory=memory,
            enable_agentic_memory=True,
            enable_user_memories=True,
            storage=storage,
            add_history_to_messages=True,
            num_history_runs=10,
            read_chat_history=True,
            description=language_instruction
        )
        
        self.agent_cache[session_id] = agent
        return agent

    async def process_health_query(self, user_input: str, session_id: str, files: List[UploadFile], preferred_language: str = 'english') -> str:
        """Processes a health query, including uploaded files and proper memory handling."""
        detected_language = preferred_language if preferred_language else self.detect_language(user_input)

        if self.check_emergency_keywords(user_input):
            emergency_response = {
                'english': "⚠️ If you're experiencing a medical emergency, please call your local emergency number immediately.",
                'hindi': "⚠️ यदि आप एक चिकित्सा आपातकाल का सामना कर रहे हैं, तो कृपया तुरंत अपने स्थानीय आपातकालीन नंबर पर कॉल करें।",
                'gujarati': "⚠️ જો તમે તબીબી કટોકટીનો સામનો કરી રહ્યા છો, તો કૃપા કરીને તાત્કાલિક તમારા સ્થાનિક કટોકટી નંબર પર કૉલ કરો।",
                'tamil': "⚠️ நீங்கள் மருத்துவ அவசரநிலையை சந்தித்தால், உடனடியாக உங்கள் உள்ளூர் அவசரகால எண்ணை அழைக்கவும்.",
                'telugu': "⚠️ మీరు వైద్య అత్యవసర పరిస్థితిని ఎదుర్కొంటున్నట్లయితే, దయచేసి వెంటనే మీ స్థానిక అత్యవసర నంబర్‌కు కాల్ చేయండి।"
            }
            return emergency_response.get(detected_language, emergency_response['english'])

        try:
            general_agent = self._create_general_agent(detected_language, session_id)
            agno_files = []
            file_names = []
            if files:
                for uploaded_file in files:
                    if uploaded_file.filename:
                        file_names.append(uploaded_file.filename)
                        content_bytes = await uploaded_file.read()
                        agno_files.append(File(content=content_bytes, filename=uploaded_file.filename))

            enhanced_input = user_input
            if agno_files:
                enhanced_input += f"\n\n[User has uploaded {len(agno_files)} file(s): {', '.join(file_names)}. Please analyze them in {detected_language.title()}.]"

            response = general_agent.run(
                enhanced_input,
                session_id=session_id,
                user_id=session_id,
                files=agno_files if agno_files else None
            )
            
            response_text = response.content if hasattr(response, 'content') else str(response)
            return response_text
        
        except Exception as e:
            print(f"An unexpected error occurred in process_health_query: {e}")
            traceback.print_exc()
            error_messages = {
                'english': "I apologize, but I encountered an error. Please try again.",
                'hindi': "मुझे खेद है, एक त्रुटि हुई। कृपया पुन: प्रयास करें।",
                'gujarati': "હું માફી માંગું છું, એક ભૂલ આવી. કૃપા કરીને ફરીથી પ્રયાસ કરો।",
                'tamil': "மன்னிக்கவும், ஒரு பிழை ஏற்பட்டது. தயவுசெய்து மீண்டும் முயற்சிக்கவும்.",
                'telugu': "క్షమించండి, లోపం వచ్చింది. దయచేసి మళ్లీ ప్రయత్నించండి."
            }
            return error_messages.get(detected_language, error_messages['english'])

    def detect_language(self, text: str) -> str:
        text_lower = text.lower()
        scores = {lang: sum(1 for pattern in patterns if pattern in text_lower) for lang, patterns in self.language_patterns.items()}
        return max(scores, key=scores.get) if max(scores.values()) > 0 else 'english'

    def check_emergency_keywords(self, text: str) -> bool:
        emergency_keywords = [
            'emergency', 'urgent', 'chest pain', 'heart attack', 'stroke', 'bleeding', 'unconscious', 
            'severe pain', 'difficulty breathing', 'suicide', 'आपातकाल', 'तुरंत', 'कटोकटी', 'તાત્કાલિક',
            'அவசரம்', 'तातडीची', 'అత్యవసర'
        ]
        return any(keyword in text.lower() for keyword in emergency_keywords)

# --- FastAPI Application ---
app = FastAPI(
    title="Satya Health Assistant API",
    description="AI-powered health assistant with multilingual support and authentication",
    version="1.0.0"
)

# CORS middleware for cross-origin requests from the React Native app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Adjust this to your specific frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
templates = Jinja2Templates(directory="templates")

health_agent = HealthAgent()

# --- Authentication Endpoints ---
@app.post("/auth/signup", response_class=JSONResponse)
async def signup(user_data: UserSignup):
    try:
        if user_db.create_user(user_data.email, user_data.password, user_data.preferred_language):
            return {"message": "User created successfully"}
        else:
            return JSONResponse(status_code=400, content={"error": "User already exists"})
    except Exception as e:
        print(f"Signup error: {e}")
        return JSONResponse(status_code=500, content={"error": "Internal server error"})

@app.post("/auth/login", response_class=JSONResponse)
async def login(user_data: UserLogin):
    try:
        print(f"=== LOGIN DEBUG ===")
        print(f"Received login attempt for email: {user_data.email}")
        print(f"Password length: {len(user_data.password)}")
        conn = sqlite3.connect(user_db.db_file)
        cursor = conn.cursor()
        cursor.execute('SELECT email, password_hash FROM users WHERE email = ?', (user_data.email,))
        user_exists = cursor.fetchone()
        conn.close()
        if not user_exists:
            print(f"❌ User not found: {user_data.email}")
            return JSONResponse(status_code=401, content={"error": "User not found"})
        print(f"✅ User found in database")
        user = user_db.authenticate_user(user_data.email, user_data.password)
        if user:
            session_token = user_db.create_session(user['email'])
            response = JSONResponse(content={"message": "Login successful", "user": user})
            # For local testing, httponly=False and secure=False is fine
            response.headers["Authorization"] = f"Bearer {session_token}"
            return response
        else:
            return JSONResponse(status_code=401, content={"error": "Invalid email or password"})
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": "Internal server error"})

@app.get("/auth/me", response_class=JSONResponse)
async def get_current_user(authorization: str = Header(None)) -> Optional[dict]:
    """Get current authenticated user from Authorization header."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    
    try:
        # The header value is expected to be "Bearer <token>"
        scheme, session_token = authorization.split(' ', 1)  # Split only on first space
        if scheme.lower() != "bearer":
            raise HTTPException(status_code=401, detail="Invalid authorization scheme")
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid authorization header format")
    
    user = user_db.get_user_by_session(session_token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired session token")
    
    return user

@app.post("/auth/logout", response_class=JSONResponse)
async def logout(request: Request):
    session_token = request.cookies.get("session_token")
    if session_token:
        user_db.delete_session(session_token)
    response = JSONResponse(content={"message": "Logged out successfully"})
    response.delete_cookie("session_token")
    return response

@app.post("/chat", response_class=JSONResponse)
async def handle_chat(
    message: str = Form(""),
    files: List[UploadFile] = Form(default=[]),  # Fixed: Use Form instead of File
    authorization: str = Header(None)
):
    # Get current user using the authorization header
    current_user = await get_current_user(authorization)
    
    try:
        # Filter for only PDF files
        pdf_files = []
        if files:
            for file in files:
                if file.filename and file.content_type == "application/pdf":
                    pdf_files.append(file)
                elif file.filename:  # Non-PDF file
                    return JSONResponse(
                        status_code=400, 
                        content={"error": f"Only PDF files are allowed. Found: {file.content_type}"}
                    )

        # Use email as the session ID for memory persistence
        session_id = current_user['email']
        preferred_language = current_user['preferred_language']

        print(f"Processing chat for user: {current_user['email']}, message: {message[:50]}...")

        response_text = await health_agent.process_health_query(
            user_input=message,
            session_id=session_id,
            files=pdf_files,
            preferred_language=preferred_language
        )

        return {
            "reply": response_text,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        print(f"Chat processing error: {e}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "error": f"An error occurred while processing your message: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
        )

# Fixed chat history endpoints with AI-generated names
@app.get("/newchats", response_model=List[Dict])
async def get_chats(authorization: str = Header(None)):
    current_user = await get_current_user(authorization)
    return user_db.get_chats_for_user(current_user['email'])

@app.post("/newchats", status_code=status.HTTP_201_CREATED)
async def create_new_chat(chat_data: ChatRequest, authorization: str = Header(None)):
    current_user = await get_current_user(authorization)
    
    # Use AI-generated name if name is generic or not provided
    if not chat_data.name or chat_data.name.strip() in ["New Chat", "", "new chat"]:
        chat_id = await user_db.create_new_chat(
            current_user['email'], 
            chat_data.messages, 
            current_user['preferred_language']
        )
    else:
        # Use provided name
        chat_id = user_db.create_new_chat_with_name(
            current_user['email'], 
            chat_data.name, 
            chat_data.messages
        )
    
    if not chat_id:
        raise HTTPException(status_code=500, detail="Failed to create new chat.")
    return {"message": "Chat created successfully", "chat_id": chat_id}

@app.put("/newchats/{chat_id}")
async def update_chat(chat_id: int, chat_data: ChatUpdate, authorization: str = Header(None)):
    current_user = await get_current_user(authorization)
    success = user_db.update_chat(chat_id, current_user['email'], chat_data.messages)
    if not success:
        raise HTTPException(status_code=404, detail="Chat not found or user not authorized.")
    return {"message": "Chat updated successfully"}

@app.get("/newchats/{chat_id}")
async def get_chat(chat_id: int, authorization: str = Header(None)):
    current_user = await get_current_user(authorization)
    chat = user_db.get_chat_by_id(chat_id, current_user['email'])
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found or user not authorized.")
    return chat

@app.post("/transcribe", response_class=JSONResponse)
async def transcribe_audio(file: UploadFile = FastAPIFile(...)):
    """Transcribe uploaded audio in any language using OpenAI Whisper."""
    if not stt_client:
        return JSONResponse(status_code=503, content={"error": "Speech-to-text service unavailable"})

    try:
        # Validate file name
        if not file.filename:
            return JSONResponse(status_code=400, content={"error": "No file provided"})

        # Read file contents
        audio_content = await file.read()
        if not audio_content:
            return JSONResponse(status_code=400, content={"error": "Empty audio file"})

        # Check size (max 25MB for Whisper)
        if len(audio_content) > 25 * 1024 * 1024:
            return JSONResponse(status_code=400, content={"error": "File too large (max 25MB)"})

        # Ensure temp dir
        os.makedirs("tmp", exist_ok=True)

        # Determine extension
        _, ext = os.path.splitext(file.filename.lower())
        ext = ext.lstrip(".") or "m4a"  # default if missing
        timestamp = int(datetime.now().timestamp())
        temp_path = f"tmp/temp_audio_{timestamp}.{ext}"

        # Save file
        with open(temp_path, "wb") as f:
            f.write(audio_content)

        print(f"Saved audio to {temp_path} ({len(audio_content)} bytes)")

        # Send to Whisper (auto language detection)
        with open(temp_path, "rb") as audio_file:
            transcript = await stt_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text",  # simple plain text
                language=None,           # let Whisper auto-detect
                temperature=0.0           # most accurate transcription
            )

        # Cleanup temp file
        try:
            os.remove(temp_path)
        except Exception as cleanup_err:
            print(f"Warning: could not delete temp file: {cleanup_err}")

        return JSONResponse(status_code=200, content={
            "transcription": transcript.strip() if isinstance(transcript, str) else str(transcript).strip(),
            "file_info": {
                "filename": file.filename,
                "content_type": file.content_type,
                "size": len(audio_content),
                "extension": ext
            },
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        print(f"Transcription error: {e}")
        traceback.print_exc()

        # Try to delete temp file if something failed
        try:
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.remove(temp_path)
        except:
            pass

        return JSONResponse(status_code=500, content={
            "error": "Failed to transcribe audio",
            "details": str(e)
        })

@app.api_route("/debug/body", methods=["GET", "POST"])
async def debug_body(request: Request):
    body_bytes = await request.body()
    try:
        body_str = body_bytes.decode("utf-8", errors="replace")
    except Exception:
        body_str = str(body_bytes)
    return {
        "headers": dict(request.headers),
        "body_raw": body_str
    }

if __name__ == "__main__":
    import uvicorn
    print("Starting Satya Health Assistant API...")
    print("Make sure to set your OpenAI API key in the .env file")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
