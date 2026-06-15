from flask import Flask, jsonify, request
from generator.llm_gen import ResponseGen
import uuid
app = Flask (__name__)

class SessionManager:
    def __init__(self):
        self.store = {}

    def create(self):
        session_id = str(uuid.uuid4())
        self.store[session_id] = ResponseGen()
        return session_id

    def get(self, session_id)->ResponseGen:
        return self.store.get(session_id, ResponseGen())

    def is_valid(self, session_id):
        return session_id in self.store

session_manager = SessionManager()

@app.route("/api/generation/health", methods=["GET"])
def health ():
    return jsonify(
        {
            "status": "ok"
        }
    ), 200

@app.route("/api/generation/chat", methods=["POST"])
def chat():
    data = request.get_json()

    query = data.get("query")
    vision_results = data.get("vision_results")
    session_id = data.get("session_id")

    # validate
    if not query:
        return jsonify({"error": "query is required"}), 400
    if not vision_results:
        return jsonify({"error": "vision_results is required"}), 400

    # create new session if needed
    if not session_id or not session_manager.is_valid(session_id):
        session_id = session_manager.create()

    # get this user's ResponseGen
    response_gen = session_manager.get(session_id)

    # generate
    result = response_gen.gen_response(query=query, vision_results=vision_results)

    return jsonify({
        "session_id": session_id,   # client must store this
        "message": result
    }), 200

if __name__ == "__main__":
    app.run(port=5002, debug=True)