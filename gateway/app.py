from flask import Flask, request, jsonify
import requests
from flask_cors import CORS

app = Flask(__name__)

host = 'http://127.0.0.1'
CORS(app)

services = {
    "vision_service": f"{host}:5001/api/vision/health",
    "generation_service": f"{host}:5002/api/generation/health"
}

from flask import send_from_directory
import os

DATASET_DIR = "/home/aritra-mukherjee/projects/mmrag/dataset/various_tagged_images"

@app.route("/images/<filename>")
def serve_image(filename):
    return send_from_directory(DATASET_DIR, filename)

@app.route("/api/health", methods=["GET"])
def health():
    status = {}
    for name, url in services.items():
        try:
            r = requests.get(url, timeout=3)
            status[name] = "ok" if r.status_code == 200 else "error"
        except:
            status[name] = "unreachable"
    return jsonify(status), 200


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    img = data.get('image')
    query = data.get('query')
    session_id = data.get('session_id')

    if not query:
        return jsonify({"error": "query is required"}), 400

    vision_result = requests.post(
        f"{host}:5001/api/vision/analyze",
        json={
            "img": img,
            "text": query,
            "has_image_attachment": img is not None
        }
    )
    vision_data = vision_result.json()

    gen_result = requests.post(
        f"{host}:5002/api/generation/chat",
        json={
            "query": query,
            "vision_results": vision_data,
            "session_id": session_id
        }
    )
    gen_data = gen_result.json()

    # extract matched image info from vision_data
    matches = []
    metadatas = vision_data.get("metadatas", [[]])[0]
    distances = vision_data.get("distances", [[]])[0]
    for meta, distance in zip(metadatas, distances):
        filename = os.path.basename(meta["url"])
        matches.append({
            "url": f"http://127.0.0.1:5000/images/{filename}",
            "caption": meta["caption"],
            "similarity": round(1 - distance, 4)
        })

    return jsonify({
        "message": gen_data["message"],
        "session_id": gen_data["session_id"],
        "matches": matches
    }), 200


if __name__ == "__main__":
    app.run(port=5000, debug=True)
    