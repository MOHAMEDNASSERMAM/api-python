import os
import sys
import httpx
from typing import Optional, Dict, Any

# Fix Arabic printing on Windows (cp1252 -> utf-8)
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import psycopg2
import psycopg2.extras
from transformers import pipeline

app = FastAPI(title="Jobito AI Chatbot (Local Generative AI)")

DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     os.getenv("DB_PORT", "5432"),
    "dbname":   os.getenv("DB_NAME", "jobito"),
    "user":     os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASS", "postgres"),
    "options":  "-c search_path=ptj,public",
}

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


print("جاري تحميل نموذج الذكاء الاصطناعي المحلي (Qwen2.5-0.5B-Instruct)...")
print("ملاحظة: هذا قد يستغرق بعض الوقت (لتحميل حوالي 1-2 جيجابايت في المرة الأولى) وسيعمل محلياً بالكامل.")

try:
    # Use HuggingFace pipeline with a small instruction model that supports Arabic heavily
    ai_pipeline = pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        device="cpu", # Defaults to CPU so it runs safely anywhere. Change to "cuda" for GPU speedup.
    )
    print("تم تحميل النموذج المحلي بنجاح! السيرفر جاهز للعمل.")
except Exception as e:
    print(f"حدث خطأ أثناء تحميل النموذج المحلي: {e}")
    ai_pipeline = None

# ═══════════════════════════════════════════════════════════════════════════════
# CONTEXT RETRIEVAL (RAG) & MEMORY
# ═══════════════════════════════════════════════════════════════════════════════

# Simple in-memory memory store: {user_id: [messages]}
chat_memory: Dict[str, list] = {}

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
    # Extract potential keywords (simple split for now)
    keywords = [w for w in query.split() if len(w) > 3]
    search_term = f"%{keywords[0]}%" if keywords else "%"
    
    sql = """
        SELECT title, salary_min, salary_max 
        FROM ptj.jobs 
        WHERE (title ILIKE %s OR description ILIKE %s) AND is_active = TRUE 
        LIMIT 3
    """
    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, (search_term, search_term))
                jobs = cur.fetchall()
                if not jobs: return ""
                res = "الوظائف المتاحة حالياً والمناسبة لطلبك: "
                res += "، ".join([f"{j['title']} (راتب: {j['salary_min']}-{j['salary_max']})" for j in jobs])
                return res
    except: return ""

def fetch_company_context(query: str):
    sql = "SELECT name, industry, description FROM companies WHERE name ILIKE %s OR description ILIKE %s LIMIT 1"
    # Try to find a specific company name if mentioned
    search_term = "%"
    words = query.split()
    if len(words) > 1: search_term = f"%{words[-1]}%"

    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, (search_term, search_term))
                c = cur.fetchone()
                if not c: return ""
                return f"معلومات عن شركة {c['name']} ({c['industry']}): {c['description'][:150]}..."
    except: return ""

def fetch_help_context(query: str):
    sql = "SELECT title, content FROM ptj.help_articles WHERE title ILIKE %s OR content ILIKE %s LIMIT 1"
    search_term = f"%{query.split()[-1]}%" if query.split() else "%"
    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, (search_term, search_term))
                h = cur.fetchone()
                if not h: return ""
                return f"إليك المساعدة بخصوص {h['title']}: {h['content'][:200]}..."
    except: return ""

class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = "guest"

@app.post("/chat")
async def chat(request: ChatRequest):
    if not request.message:
        raise HTTPException(status_code=400, detail="Empty message")
    
    if not ai_pipeline:
        raise HTTPException(status_code=500, detail="عذراً، نظام الذكاء الاصطناعي المحلي غير متاح حالياً.")

    user_id = request.user_id or "guest"
    user_msg = request.message
    
    # 1. Retrieve Context from Database
    db_context = get_db_context(user_msg.lower())

    # 2. Manage Memory (Last 4 rounds)
    if user_id not in chat_memory:
        chat_memory[user_id] = []
    
    # 3. Build Prompt with History
    sys_prompt = "أنت مساعد ذكي لمنصة Jobito. أجِب بأسلوب عربي ودود ومختصر. استخدم المعلومات المتاحة من قاعدة البيانات فقط للرد بدقة."
    if db_context:
        sys_prompt += f"\n\nمعلومات من النظام:\n{db_context}"

    messages = [{"role": "system", "content": sys_prompt}]
    
    # Add history
    for old_msg in chat_memory[user_id][-4:]:
        messages.append(old_msg)
    
    # Add current message
    messages.append({"role": "user", "content": user_msg})

    # 4. Ask Local Generative AI
    try:
        result = ai_pipeline(
            messages,
            max_new_tokens=200,
            temperature=0.6,
            do_sample=True,
        )
        reply = result[0]["generated_text"][-1]["content"]
        
        # Save to memory
        chat_memory[user_id].append({"role": "user", "content": user_msg})
        chat_memory[user_id].append({"role": "assistant", "content": reply})
        # Keep memory short
        if len(chat_memory[user_id]) > 10: chat_memory[user_id] = chat_memory[user_id][-10:]

    except Exception as e:
        await report_to_bam(f"Local AI Error: {str(e)}")
        reply = "عذراً، واجهت مشكلة في التفكير. حاول مرة أخرى."
        print(f"[Error]: {e}")

    return {"reply": reply}

if __name__ == "__main__":
    import uvicorn
    # Use multiple workers carefully; loading LLM in memory per worker requires significant RAM
    uvicorn.run(app, host="0.0.0.0", port=5000)
