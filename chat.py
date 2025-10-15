from flask import Flask, send_from_directory
from flask import request, jsonify
from flask_cors import CORS
import webbrowser
import requests
import argparse
from utils import *


app = Flask(__name__, static_folder=".")
CORS(app)

parser = argparse.ArgumentParser()
parser.add_argument("--host", type=str, default="10.208.60.137")
parser.add_argument("--port", type=str, default="4007")
parser.add_argument("--max_tokens", type=int, default=256)
parser.add_argument("--load", type=str, default=None)

args = parser.parse_args()

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data or "messages" not in data:
        return jsonify({"reply": "no reaction"})
    
    host = args.host
    port = args.port
    url = f"http://{host}:{port}/v1/chat/completions"
    
    messages = data["messages"]
    payload = {
        "model": "",
        "messages": messages,
        "temperature": 0,
        "max_tokens": args.max_tokens
    }

    try:
        response = requests.post(url, json=payload, timeout=60)
        response_json = response.json()
        prediction = response_json["choices"][0]["message"]["content"]
        return jsonify({"reply": prediction})

    except Exception as e:
        return jsonify({"reply": "no reaction"})

@app.route("/init", methods=["GET"])
def init():
    initial_messages = read_json(args.load) if args.load else []
    return jsonify({"messages": initial_messages})


if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:5000/")
    app.run(debug=False)
