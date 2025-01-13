# app.py

import logging
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from pipeline import pipeline
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os
import asyncio  # Ensure asyncio is imported

# Ensure the logs directory exists
if not os.path.exists('logs'):
    os.makedirs('logs')

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)

app = Flask(__name__)
CORS(app)

# Initialize Limiter for rate limiting
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
async def chat():  # Make the route asynchronous
    data = request.get_json()
    if not data or 'prompt' not in data:
        logging.warning("No prompt found in the request.")
        return jsonify({"status": "error", "error": "No prompt provided."}), 400

    prompt = data.get("prompt", "")
    logging.info(f"Received prompt: {prompt}")

    try:
        # Await the async pipeline function directly
        response = await pipeline(prompt)
        logging.info(f"Pipeline Result: {response}")

        if response.get("is_safe"):
            return jsonify({
                "status": "allowed",
                "response": response.get("response", "No response generated."),
                "checks": response.get("checks", [])
            })
        else:
            return jsonify({
                "status": "blocked",
                "error": response.get("error", "Blocked due to safety concerns."),
                "checks": response.get("checks", [])
            }), 403

    except Exception as e:
        logging.error(f"Error in processing: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500

if __name__ == "__main__":
    logging.info("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True)
