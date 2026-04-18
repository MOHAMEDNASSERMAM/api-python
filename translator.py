from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from deep_translator import GoogleTranslator
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

app = FastAPI(title="Jobito Stateless Translation Service")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared thread pool for parallel translations
executor = ThreadPoolExecutor(max_workers=10)

# Pre-instantiated translators for common pairs
translators = {
    "en_ar": GoogleTranslator(source='en', target='ar'),
    "ar_en": GoogleTranslator(source='ar', target='en'),
    "auto_en": GoogleTranslator(source='auto', target='en'),
    "auto_ar": GoogleTranslator(source='auto', target='ar'),
}

class TranslateRequest(BaseModel):
    text: Optional[str] = None
    texts: Optional[List[str]] = None
    source_lang: str = "auto"
    target_lang: str = "en"

@lru_cache(maxsize=1024)
def _get_translator(source, target):
    key = f"{source}_{target}"
    translator = translators.get(key)
    if not translator:
        translator = GoogleTranslator(source=source, target=target)
    return translator

@lru_cache(maxsize=2048)
def translate_single_sync(text: str, source: str, target: str):
    if not text or not text.strip():
        return text
        
    translator = _get_translator(source, target)
    try:
        return translator.translate(text)
    except Exception as e:
        print(f"⚠️ Translation error for '{text[:20]}...': {e}")
        return text

@app.get("/health")
def health():
    return {"status": "ok", "service": "translation-stateless"}

@app.post("/translate")
async def translate(req: TranslateRequest):
    start_time = time.time()
    
    source = req.source_lang
    target = req.target_lang
    
    # Sanitize inputs
    texts = req.texts if req.texts else ([req.text] if req.text else [])
    if not texts:
        raise HTTPException(status_code=400, detail="No text provided")

    print(f"📥 [Batch Request]: {len(texts)} items | {source} -> {target}")

    try:
        # Use native batch translation for massive speed boost and to stop Google from blocking the IP
        # We replace empty strings with a space as deep_translator batching requires valid strings
        safe_texts = [t if (t and t.strip()) else " " for t in texts]
        translator = _get_translator(source, target)
        
        # Deep_translator's batch translation uses optimized grouping
        raw_results = await asyncio.to_thread(translator.translate_batch, safe_texts)
        
        # Restore empty strings if any
        results = [r.strip() if r and r.strip() else text for r, text in zip(raw_results, texts)]
                

        latency = time.time() - start_time
        print(f"✅ Success | Parallel Latency: {latency:.2f}s")
        
        if req.text:
            return {"translated_text": results[0], "latency": f"{latency:.4f}s"}
        
        return {
            "translated_texts": results,
            "count": len(results),
            "latency": f"{latency:.4f}s"
        }

    except Exception as e:
        print(f"❌ Critical Translation Error: {e}")
        # Fallback: return original texts if everything fails
        fallback_results = texts if req.texts else [req.text]
        if req.text:
            return {"translated_text": fallback_results[0], "error": str(e)}
        return {"translated_texts": fallback_results, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    import sys
    print("🚀 Starting Optimized Translation service on port 5001...")
    try:
        # Run with multiple workers for concurrency, though ThreadPool handles the CPU/IO wait
        uvicorn.run(app, host="0.0.0.0", port=5001, log_level="info")
    except KeyboardInterrupt:
        print("\n👋 Service stopped by user.")
        sys.exit(0)
    except Exception as e:
        print(f"💥 Failed to start server: {e}")
        sys.exit(1)