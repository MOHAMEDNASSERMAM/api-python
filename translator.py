from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from deep_translator import GoogleTranslator
import time

app = FastAPI(title="Jobito Stateless Translation Service")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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

def translate_single(text: str, source: str, target: str):
    if not text:
        return ""
        
    key = f"{source}_{target}"
    translator = translators.get(key)
    if not translator:
        translator = GoogleTranslator(source=source, target=target)
    
    try:
        return translator.translate(text)
    except Exception as e:
        print(f"Translation error: {e}")
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
        key = f"{source}_{target}"
        translator = translators.get(key)
        if not translator:
            translator = GoogleTranslator(source=source, target=target)
            translators[key] = translator
        
        # Limit batch size to prevent Google API blocks or timeouts
        MAX_BATCH = 25
        results = []
        
        for i in range(0, len(texts), MAX_BATCH):
            batch = [t for t in texts[i:i+MAX_BATCH] if t and t.strip()]
            if not batch:
                results.extend([""] * (min(i + MAX_BATCH, len(texts)) - i))
                continue
                
            try:
                translated_batch = translator.translate_batch(batch)
                results.extend(translated_batch)
            except Exception as batch_err:
                print(f"⚠️ Chunk translation error: {batch_err}. Falling back to single mode.")
                for individual_text in batch:
                    results.append(translate_single(individual_text, source, target))
        
        # Ensure results match input length
        while len(results) < len(texts):
            results.append(texts[len(results)])

        print(f"✅ Success | Latency: {time.time() - start_time:.2f}s")
        
        if req.text:
            return {"translated_text": results[0], "latency": f"{time.time() - start_time:.4f}s"}
        
        return {
            "translated_texts": results,
            "count": len(results),
            "latency": f"{time.time() - start_time:.4f}s"
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
    print("🚀 Starting Resilient Translation service on port 5001...")
    try:
        uvicorn.run(app, host="0.0.0.0", port=5001, log_level="info")
    except KeyboardInterrupt:
        print("\n👋 Service stopped by user.")
        sys.exit(0)
    except Exception as e:
        print(f"💥 Failed to start server: {e}")
        sys.exit(1)