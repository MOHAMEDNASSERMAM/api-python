import os
import json
import re
import httpx
import asyncio
from datetime import datetime
from collections import defaultdict
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import psycopg2
import psycopg2.extras

app = FastAPI(title="Jobito Internal AI Chatbot")

DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     os.getenv("DB_PORT", "5432"),
    "dbname":   os.getenv("DB_NAME", "jobito"),
    "user":     os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASS", "postgres"),
    "options":  "-c search_path=ptj,public",
}

# Local Monitoring Endpoint (NestJS)
NESTJS_MONITORING_URL = os.getenv("NESTJS_MONITORING_URL", "http://localhost:3000/monitoring/log")

# ═══════════════════════════════════════════════════════════════════════════════
# MONITORING BRIDGE (BAM) - Local Only
# ═══════════════════════════════════════════════════════════════════════════════

async def report_to_bam(message: str, metadata: Dict[str, Any] = None):
    """Reports internal chatbot events/errors to the NestJS monitoring system."""
    try:
        async with httpx.AsyncClient() as client:
            payload = {
                "message": f"[ChatBot-Internal] {message}",
                "metadata": metadata or {}
            }
            await client.post(NESTJS_MONITORING_URL, json=payload, timeout=1.0)
    except:
        pass # Don't crash if monitoring is down

INTENT_PATTERNS = {
    "search_jobs": [
        r"وظيفة", r"شغل", r"بحث", r"دور", r"عايز", r"فرصة",
        r"job", r"work", r"search", r"find", r"vacancy"
    ],
    "company_info": [
        r"شركة", r"شركات", r"معلومات", r"مين",
        r"company", r"companies", r"about", r"who"
    ],
    "application_status": [
        r"طلبي", r"تقديم", r"حالة", r"وصل", r"قدمت",
        r"status", r"applied", r"my application", r"tracking"
    ],
    "platform_stats": [
        r"أرقام", r"إحصائيات", r"كم", r"عدد",
        r"stats", r"statistics", r"how many", r"count"
    ],
    "help": [
        r"مساعدة", r"تفعيل", r"مشكلة", r"ازاي",
        r"help", r"support", r"how to", r"problem"
    ],
    "greeting": [
        r"سلام", r"مرحبا", r"أهلا", r"هاي", r"صباح", r"مساء",
        r"hi", r"hello", r"hey", "welcome"
    ]
}

def analyze_intent_locally(message: str) -> Dict[str, Any]:
    """Purely local intent detection without any external calls."""
    msg = message.lower().strip()
    detected_intent = "unknown"
    
    # 1. Intent Detection
    for intent, patterns in INTENT_PATTERNS.items():
        if any(re.search(p, msg) for p in patterns):
            detected_intent = intent
            break
            
    # 2. Basic Entity Extraction
    filters = {
        "keyword": None,
        "job_type": None,
        "location": None
    }
    
    # Simple extraction logic (can be expanded)
    if "في" in msg:
        parts = msg.split("في")
        if len(parts) > 1:
            filters["location"] = parts[1].strip()
            
    # 3. Language detection
    lang = "ar" if any("\u0600" <= c <= "\u06FF" for c in message) else "en"
    
    return {
        "intent": detected_intent,
        "filters": filters,
        "language": lang
    }

def get_connection():
    return psycopg2.connect(**DB_CONFIG)

async def query_db(sql: str, params: tuple = None, fetchone: bool = False):
    conn = None
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params or ())
            if cur.description:
                return dict(cur.fetchone()) if fetchone else [dict(r) for r in cur.fetchall()]
            return []
    except Exception as e:
        await report_to_bam(f"DB Error: {str(e)}", {"sql": sql})
        return []
    finally:
        if conn: conn.close()

async def handle_request(analysis: dict, user_id: str) -> str:
    intent = analysis["intent"]
    lang = analysis["language"]
    
    if intent == "greeting":
        return "أهلاً بك في Jobito! كيف يمكنني مساعدتك اليوم؟" if lang == "ar" else "Hello! How can I help you today?"
        
    if intent == "search_jobs":
        sql = "SELECT title, title_en FROM jobs WHERE is_active = TRUE LIMIT 3"
        rows = await query_db(sql)
        if not rows: return "عذراً، لم أجد وظائف حالياً." if lang == "ar" else "No jobs found."
        titles = [r["title"] or r["title_en"] for r in rows]
        return "إليك بعض الوظائف: " + ", ".join(titles)
        
    if intent == "platform_stats":
        total_jobs = await query_db("SELECT COUNT(*) FROM jobs", fetchone=True)
        count = total_jobs.get("count", 0)
        return f"يوجد لدينا حالياً {count} وظيفة نشطة." if lang == "ar" else f"We have {count} active jobs."

    return "أنا مساعدك الذكي لـ Jobito، يمكنك سؤالي عن الوظائف أو الشركات." if lang == "ar" else "I can help you with jobs or companies."

class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = "guest"

@app.post("/chat")
async def chat(request: ChatRequest):
    if not request.message:
        raise HTTPException(status_code=400, detail="Empty message")
        
    analysis = analyze_intent_locally(request.message)
    
    reply = await handle_request(analysis, request.user_id)
    return {"reply": reply}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
