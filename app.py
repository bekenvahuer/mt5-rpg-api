from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import requests
import json

app = Flask(__name__)
CORS(app)

# === Configuración del modelo en Hugging Face ===
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "bekenRey/mt5-small-rpg-mission-generator-english")
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"

HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

@app.get("/")
def health():
    return {"ok": True, "model": HF_MODEL_ID, "service": "RPG Mission Generator API"}

@app.get("/api/health")
def health_check():
    try:
        resp = requests.get(f"https://huggingface.co/api/models/{HF_MODEL_ID}", timeout=10)
        status = "available" if resp.status_code == 200 else "unavailable"
        return jsonify({
            "status": "healthy", 
            "model": HF_MODEL_ID,
            "hf_status": status,
            "mode": "hf_inference_api"
        })
    except:
        return jsonify({"status": "healthy", "model": HF_MODEL_ID, "hf_status": "unknown"})

@app.post("/api/mission")
def generar_mision():
    data = request.get_json(force=True) or {}
    prompt = data.get("prompt", "Generate a fantasy RPG mission")
    
    formatted_prompt = f"generate mission: {prompt}"

    payload = {
        "inputs": formatted_prompt,
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.9,
            "top_k": 50,
            "top_p": 0.95,
            "do_sample": True,
            "return_full_text": False
        }
    }

    try:
        resp = requests.post(HF_API_URL, json=payload, headers=HEADERS, timeout=60)
        
        if resp.status_code == 503:
            estimated_time = resp.json().get("estimated_time", 30)
            return jsonify({
                "error": "Model is loading", 
                "estimated_time": estimated_time,
                "retry_after": estimated_time
            }), 503

        resp.raise_for_status()
        out = resp.json()

    except requests.exceptions.Timeout:
        return jsonify({"error": "HF API timeout"}), 504
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"HF API error: {str(e)}"}), 500

    # Extraer el texto generado
    if isinstance(out, list) and out:
        if isinstance(out[0], dict) and "generated_text" in out[0]:
            generated_text = out[0]["generated_text"]
        else:
            generated_text = str(out[0])
    elif isinstance(out, dict) and "generated_text" in out:
        generated_text = out["generated_text"]
    else:
        generated_text = str(out)

    # Intentar parsear como JSON
    try:
        parsed_json = json.loads(generated_text)
        return jsonify(parsed_json)
    except json.JSONDecodeError:
        return jsonify({"generated_text": generated_text})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)