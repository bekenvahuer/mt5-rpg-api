from flask import Flask, jsonify, request
from flask_cors import CORS
import json
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = Flask(__name__)
CORS(app)

# ====== CONFIGURACIÓN DEL MODELO ======
MODEL_NAME = "bekenRey/mt5-small-rpg-mission-generator-english"

# Cargar modelo (con manejo de errores para Render)
try:
    print("⏳ Cargando modelo...")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"✅ Modelo cargado en {device}")
except Exception as e:
    print(f"❌ Error cargando modelo: {e}")
    tokenizer, model = None, None

@app.route('/')
def home():
    return {"status": "online", "model": MODEL_NAME}

@app.route('/api/health', methods=['GET'])
def health_check():
    if model and tokenizer:
        return jsonify({"status": "healthy", "model": MODEL_NAME, "device": str(device)})
    else:
        return jsonify({"status": "error", "message": "Model not loaded"}), 500

@app.route('/api/mission', methods=['POST'])
def generar_mision():
    if not model or not tokenizer:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json()
        prompt = data.get("prompt", "Generate a fantasy mission")
        
        # Tokenizar y generar
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=128)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return jsonify({"generated_text": generated_text})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port)