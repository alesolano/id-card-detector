from flask import Flask, request, jsonify

from run import Run

app = Flask(__name__)

run = Run()

@app.route('/predict', methods=['GET'])
def predict():
    video_name = request.args.get('video_name')
    resize = request.args.get('resize')
    if resize: resize = float(resize)
    boxes = run.get_bboxes_from_video(f'/workspace/videos/{video_name}', resize=resize)
    return jsonify(boxes)


if __name__ == '__main__':
    app.run(port=5002, host='0.0.0.0')