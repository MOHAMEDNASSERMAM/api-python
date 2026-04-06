from flask import Flask, request, jsonify
from flask_cors import CORS
from deep_translator import GoogleTranslator
import traceback

app = Flask(__name__)
CORS(app)

# Cache translators to avoid re-instantiating constantly
translator_en_ar = GoogleTranslator(source='en', target='ar')
translator_ar_en = GoogleTranslator(source='ar', target='en')
translator_auto_en = GoogleTranslator(source='auto', target='en')
translator_auto_ar = GoogleTranslator(source='auto', target='ar')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "service": "translation-microservice"})

@app.route('/translate', methods=['POST'])
def translate():
    try:
        data = request.get_json(force=True)
        text = data.get('text', '').strip()
        source_lang = data.get('source_lang', 'auto')
        target_lang = data.get('target_lang', 'en')

        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Optimization for specific common pairs
        if source_lang == 'en' and target_lang == 'ar':
            translated_text = translator_en_ar.translate(text)
        elif source_lang == 'ar' and target_lang == 'en':
            translated_text = translator_ar_en.translate(text)
        elif source_lang == 'auto' and target_lang == 'en':
            translated_text = translator_auto_en.translate(text)
        elif source_lang == 'auto' and target_lang == 'ar':
            translated_text = translator_auto_ar.translate(text)
        else:
            # Fallback for dynamic pairing
            custom_translator = GoogleTranslator(source=source_lang, target=target_lang)
            translated_text = custom_translator.translate(text)

        return jsonify({
            "original_text": text,
            "translated_text": translated_text,
            "source_lang": source_lang,
            "target_lang": target_lang
        })
    except Exception as e:
        print(f"[TRANSLATE ERROR]: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Jobito Translation Microservice on port 5001...")
    # Run on port 5001 to avoid conflicts with other services
    app.run(host='0.0.0.0', port=5001, debug=False)
