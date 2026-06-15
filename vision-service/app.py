from flask import Flask, jsonify, request
from pydantic import BaseModel, ValidationError
from vision.image_encoder import VisionServiceHelper
from typing import Optional
vision_service_helper = VisionServiceHelper()

print(vision_service_helper.query_by_image_test('/home/aritra-mukherjee/projects/mmrag/dataset/test/free-nature-images.jpg'))

app = Flask(__name__)

class QueryArguments(BaseModel):
    img: Optional[str] = None
    text: Optional[str] = None
    has_image_attachment: bool = False

@app.route('/api/vision/health', methods=['GET'])
def health ():
    return jsonify(
        {
            "status": "ok"
        }
    ), 200


@app.route('/api/vision/analyze', methods=["POST"])
def analyze ():
    try:
        data = QueryArguments(**request.json)
    except ValidationError as e:
        return jsonify({
            "error": str(e)
        }), 415
    
    results = {}
    if data.has_image_attachment and data.text != None:
        results = vision_service_helper.query_by_image_and_text(data.img, data.text)
    elif data.has_image_attachment and data.text == None:
        results = vision_service_helper.query_by_image(data.img)
    elif not data.has_image_attachment and data.text != None:
        results = vision_service_helper.query_by_text(data.text)
    else:
        return jsonify({
            "error": "Invalid arguments"
        }), 400

    return jsonify(results), 200


if __name__ == "__main__":
    app.run (port = 5001, debug=True)