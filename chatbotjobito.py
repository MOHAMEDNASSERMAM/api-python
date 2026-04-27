import os
import sys
import httpx
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables from the API project
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "jobito-api", ".env"))
print(f"🔍 Loading environment from: {env_path}")
if os.path.exists(env_path):
    load_dotenv(env_path)
    print("✅ .env file found and loaded.")
else:
    print("⚠️ .env file NOT found in jobito-api folder.")


# Fix Arabic printing on Windows (cp1252 -> utf-8)
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import psycopg2
import psycopg2.extras
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
from fastapi.responses import StreamingResponse
import json
import base64
import io
from PIL import Image
import fitz # PyMuPDF
import docx

app = FastAPI(title="Jobito AI Chatbot (Local Generative AI)")

DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     os.getenv("DB_PORT", "5432"),
    "dbname":   os.getenv("DB_NAME", "jobito"),
    "user":     os.getenv("DB_USERNAME", "postgres"),
    "password": os.getenv("DB_PASSWORD", "mlpoknbv"),
    "options":  "-c search_path=ptj,public",
}

print(f"📡 DB Config: Host={DB_CONFIG['host']}, User={DB_CONFIG['user']}, DB={DB_CONFIG['dbname']}, Pass={'***' if DB_CONFIG['password'] else 'MISSING'}")


NESTJS_MONITORING_URL = os.getenv("NESTJS_MONITORING_URL", "http://localhost:3000/monitoring/log")


async def report_to_bam(message: str, metadata: Dict = None):
    try:
        async with httpx.AsyncClient() as client:
            await client.post(NESTJS_MONITORING_URL, json={
                "message": f"[ChatBot-LocalAI] {message}",
                "metadata": metadata or {}
            }, timeout=1.0)
    except:
        pass

def get_connection():
    return psycopg2.connect(**DB_CONFIG)


print("جاري تحميل نموذج النصوص الذكي (Qwen2.5)...")

try:
    # Use 0.5B model instead of 1.5B for better performance on consumer hardware
    model_id = "Qwen/Qwen2.5-0.5B-Instruct" 
    print(f"🔄 Downloading/Loading model: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to("cpu").eval()
    print("✅ تم تحميل نموذج النصوص بنجاح! السيرفر جاهز.")
except Exception as e:
    print(f"❌ فشل تحميل النموذج: {e}")
    model = None
    tokenizer = None

# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE CONNECTION TEST
# ═══════════════════════════════════════════════════════════════════════════════
def test_connections():
    # PostgreSQL Test
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                print("✅ تم الاتصال بـ PostgreSQL بنجاح.")
    except Exception as e:
        print(f"❌ فشل الاتصال بـ PostgreSQL: {e}")

test_connections()

# ═══════════════════════════════════════════════════════════════════════════════
# CONTEXT RETRIEVAL (RAG) & MEMORY
# ═══════════════════════════════════════════════════════════════════════════════

# Removed in-memory chat_memory, using MongoDB instead

def get_db_context(user_msg: str) -> str:
    """Intelligently decides what to fetch from DB based on intent."""
    context_parts = []
    
    # 1. Job search intent
    if any(k in user_msg for k in ["وظيفة", "وظائف", "شغل", "اعمل", "job", "work", "career"]):
        context_parts.append(fetch_jobs_context(user_msg))
    
    # 2. Company search intent
    if any(k in user_msg for k in ["شركة", "شركات", "company", "info about"]):
        context_parts.append(fetch_company_context(user_msg))

    # 3. Help/FAQ intent
    if any(k in user_msg for k in ["كيف", "مساعدة", "help", "how to", "مشكلة"]):
        context_parts.append(fetch_help_context(user_msg))

    return "\n".join([p for p in context_parts if p])

def fetch_jobs_context(query: str):
    # Extract potential keywords (handle Arabic better: 3+ chars)
    # Removing common stop words manually for better accuracy
    stop_words = ["اريد", "ابحث", "عن", "ما", "هي", "ممكن", "أين", "في", "على"]
    keywords = [w for w in query.split() if len(w) >= 3 and w not in stop_words]
    
    if not keywords:
        # If no keywords but they asked about jobs generally
        sql = "SELECT title, salary_min, salary_max FROM ptj.jobs WHERE is_active = TRUE ORDER BY created_at DESC LIMIT 3"
        params = ()
    else:
        search_term = f"%{keywords[0]}%"
        sql = """
            SELECT title, salary_min, salary_max 
            FROM ptj.jobs 
            WHERE (title ILIKE %s OR description ILIKE %s) AND is_active = TRUE 
            ORDER BY created_at DESC
            LIMIT 3
        """
        params = (search_term, search_term)
        
    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, params)
                jobs = cur.fetchall()
                if not jobs: 
                    return "لا توجد وظائف مطابقة تماماً حالياً، لكن يمكنك تصفح الموقع للمزيد."
                res = "الوظائف المتاحة حالياً: "
                res += " | ".join([f"{j['title']} (راتب متوقع: {int(j['salary_min'] or 0)}-{int(j['salary_max'] or 0)})" for j in jobs])
                return res
    except Exception as e: 
        print(f"DB Error (jobs): {e}")
        return ""

def fetch_company_context(query: str):
    # Clean query to find company name
    words = [w for w in query.split() if len(w) >= 3]
    search_term = f"%{words[-1]}%" if words else "%"

    sql = "SELECT name, industry, description FROM ptj.companies WHERE name ILIKE %s OR description ILIKE %s LIMIT 1"
    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, (search_term, search_term))
                c = cur.fetchone()
                if not c: return ""
                return f"معلومات عن شركة {c['name']} ({c['industry'] or 'غير محدد'}): {c['description'][:150]}..."
    except Exception as e: 
        print(f"DB Error (company): {e}")
        return ""

def fetch_help_context(query: str):
    words = [w for w in query.split() if len(w) >= 3]
    search_term = f"%{words[-1]}%" if words else "%"
    
    sql = "SELECT title, content FROM ptj.help_articles WHERE title ILIKE %s OR content ILIKE %s LIMIT 1"
    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, (search_term, search_term))
                h = cur.fetchone()
                if not h: return ""
                return f"إليك المساعدة بخصوص {h['title']}: {h['content'][:200]}..."
    except Exception as e: 
        print(f"DB Error (help): {e}")
        return ""

class ChatRequest(BaseModel):
    message: str = None 
    user_id: str = None
    history: list = None
    image: str = None # Base64 image or File content
    file_type: str = "image" # "image", "pdf", "docx"

@app.post("/chat")
async def chat(request: ChatRequest):
    if not model or not tokenizer:
        return StreamingResponse(iter([f"data: {json.dumps({'text': 'عذراً، نظام الذكاء الاصطناعي غير جاهز حالياً (فشل تحميل النموذج). يرجى التأكد من الاتصال بالإنترنت.'})}\n\n", "data: [DONE]\n\n"]), media_type="text/event-stream")

    if not request.message and not request.image:
        raise HTTPException(status_code=400, detail="Empty request")
    
    if not model or not tokenizer:
        raise HTTPException(status_code=500, detail="عذراً، نظام الذكاء الاصطناعي المحلي غير متاح حالياً.")

    user_id = request.user_id or "guest"
    user_msg = request.message
    history = request.history or []
    image_data = request.image
    f_type = request.file_type

    # 1. Handle File Processing (PDF/DOCX)
    extracted_text = ""
    pil_image = None

    if image_data:
        try:
            if "base64," in image_data:
                image_data = image_data.split("base64,")[1]
            raw_bytes = base64.b64decode(image_data)
            
            if f_type == "pdf":
                doc = fitz.open(stream=raw_bytes, filetype="pdf")
                for page in doc:
                    extracted_text += page.get_text()
                doc.close()
            elif f_type == "docx":
                doc = docx.Document(io.BytesIO(raw_bytes))
                extracted_text = "\n".join([p.text for p in doc.paragraphs])
            else:
                # Default to Image
                pil_image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        except Exception as e:
            print(f"File process error: {e}")

    # 2. Build Prompt Context
    db_context = ""
    if user_msg:
        db_context = get_db_context(user_msg.lower())

    sys_prompt = "أنت مساعد ذكي لمنصة Jobito. أجِب بأسلوب عربي ودود ومختصر.\n" \
                 "يمكنك تغيير ألوان الواجهة إذا طلب المستخدم ذلك عن طريق كتابة [THEME: color_name] في نهاية ردك.\n" \
                 "الألوان المتاحة: (dark, blue, purple, green, gold)."

    # 3. Generate Response
    def generate_chunks():
        try:
            # Proper Qwen ChatML format
            full_prompt = f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
            for msg in history[-5:]:
                role = "assistant" if msg["role"] == "assistant" else "user"
                full_prompt += f"<|im_start|>{role}\n{msg['content']}<|im_end|>\n"
            full_prompt += f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
            
            model_inputs = tokenizer([full_prompt], return_tensors="pt").to(model.device)
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            generation_kwargs = dict(model_inputs, streamer=streamer, max_new_tokens=512)
            
            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()
            
            # Add explicit stop strings to prevent token leaks
            stop_tokens = ["<|im_end|>", "<|user|>", "<|im_start|>"]
            
            # Use a specialized streamer or check for stop strings in chunks
            for new_text in streamer:
                if new_text:
                    # Break if any stop token appears in the stream
                    if any(stop in new_text for stop in stop_tokens):
                        break
                        
                    clean_text = new_text
                    # Aggressive Filtering for partial technical tokens
                    forbidden = ["<|im_start|>", "<|im_end|>", "<|user|>", "<|assistant|>", "<|system|>", "im_start", "im_end"]
                    for token in forbidden:
                        clean_text = clean_text.replace(token, "")
                    
                    if clean_text.strip() or " " in new_text:
                        yield f"data: {json.dumps({'text': clean_text})}\n\n"
            
            yield "data: [DONE]\n\n"
        except Exception as e:
            print(f"Generation error: {e}")
            yield f"data: {json.dumps({'text': 'عذراً، واجهت مشكلة في معالجة طلبك.'})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(generate_chunks(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    # Use multiple workers carefully; loading LLM in memory per worker requires significant RAM
    uvicorn.run(app, host="0.0.0.0", port=5000)
