from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import json

app = Flask(__name__)
CORS(app)

# === CONFIGURACIÓN DIRECTA DEL MODELO ===
MODEL_NAME = "bekenRey/mt5-small-rpg-mission-generator-english"

# Cargar modelo y tokenizador
try:
    print("⏳ Cargando modelo directamente...")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    
    # Optimizar para Render Free
    device = torch.device("cpu")
    model.to(device)
    model.eval()
    model.config.use_cache = False
    
    print("✅ Modelo cargado en CPU")
    
except Exception as e:
    print(f"❌ Error cargando modelo: {e}")
    tokenizer, model = None, None

@app.get("/")
def health():
    return {"ok": True, "model": MODEL_NAME, "service": "RPG Mission Generator API"}

@app.get("/api/health")
def health_check():
    if model and tokenizer:
        return jsonify({
            "status": "healthy", 
            "model": MODEL_NAME,
            "device": str(device),
            "mode": "direct_loading"
        })
    else:
        return jsonify({"status": "error", "message": "Model not loaded"}), 500

@app.post("/api/mission")
def generar_mision():
    if not model or not tokenizer:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json(force=True) or {}
    prompt = data.get("prompt", "Generate a fantasy RPG mission")
    
    try:
        # Tokenizar
        input_text = f"generate mission: {prompt}"
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        
        # Generar
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                top_k=30,
                top_p=0.9
            )
        
        # Decodificar
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Intentar parsear JSON
        try:
            parsed_json = json.loads(generated_text)
            return jsonify(parsed_json)
        except json.JSONDecodeError:
            return jsonify({"generated_text": generated_text})
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)