import logging
import os
import asyncio
from flask import Flask, request, render_template, jsonify, session
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from pipeline import pipeline

# Ensure logs directory exists
if not os.path.exists('logs'):
    os.makedirs('logs')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)

app = Flask(__name__)
app.secret_key = "replace_with_a_strong_secret_key"  # For session management
CORS(app)

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/chat", methods=["POST"])
@limiter.limit("10 per minute")
async def chat():
    data = request.get_json()
    if not data or 'prompt' not in data:
        logging.warning("No prompt found in the request.")
        return jsonify({"status": "error", "error": "No prompt provided."}), 400

    prompt = data.get("prompt", "")
    # Retrieve conversation history (if any) from the request.
    history = data.get("history", None)
    logging.info(f"Received prompt: {prompt[:50]}...")

    try:
        response = await pipeline(prompt, context=history)
        logging.info(f"Pipeline Result: {response}")
        if response.get("is_safe"):
            return jsonify({
                "status": "allowed",
                "response": response.get("response", "No response generated."),
                "checks": response.get("checks", []),
                "latency": response.get("latency", None),
                "composite_score": response.get("composite_score", None)
            })
        else:
            return jsonify({
                "status": "blocked",
                "error": response.get("error", "Blocked due to safety concerns."),
                "checks": response.get("checks", []),
                "latency": response.get("latency", None),
                "composite_score": response.get("composite_score", None)
            }), 403
    except Exception as e:
        logging.error(f"Error processing prompt: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500

if __name__ == "__main__":
    logging.info("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True)
