from flask import Flask, request, jsonify
import requests
app = Flask(__name__)

host = 'http://127.0.0.1'

services = {
    "vision_service": f"{host}:5001/api/vision/health",
    "generation_service": f"{host}:5002/api/generation/health"
}

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

    # Step 1 - Vision Service
    vision_result = requests.post(
        f"{host}:5001/api/vision/analyze",
        json={
            "img": img,
            "text": query,
            "has_image_attachment": img is not None
        }
    )

    # Step 2 - Generation Service
    gen_result = requests.post(
        f"{host}:5002/api/generation/chat",
        json={
            "query": query,
            "vision_results": vision_result.json(),
            "session_id": session_id
        }
    )

    # Step 3 - Return to client
    gen_data = gen_result.json()
    return jsonify({
        "message": gen_data["message"],
        "session_id": gen_data["session_id"]
    }), 200


if __name__ == "__main__":
    app.run(port=5000, debug=True)
    