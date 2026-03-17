"""
Target Management HTTP API Server
Port 5000, provides upload/delete/reload for target templates.

Standalone:
  python target_server.py

API:
  POST   /api/targets/upload  - Upload new target image (multipart, field="image")
  POST   /api/targets/reload  - Trigger reload
  GET    /api/targets         - List current targets
  DELETE /api/targets/<name>  - Delete a target
"""

import os
import json
import time
from flask import Flask, request, jsonify

app = Flask(__name__)

TARGETS_DIR = os.environ.get("TARGETS_DIR", "targets")
INFO_FILE = "target_info.json"

_recognizer = None


def set_recognizer(rec):
    global _recognizer
    _recognizer = rec


def _read_info():
    info_path = os.path.join(TARGETS_DIR, INFO_FILE)
    if not os.path.exists(info_path):
        return []
    with open(info_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_info(annotations):
    info_path = os.path.join(TARGETS_DIR, INFO_FILE)
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)


@app.route("/api/targets", methods=["GET"])
def list_targets():
    annotations = _read_info()
    return jsonify({"targets": annotations, "count": len(annotations)})


@app.route("/api/targets/upload", methods=["POST"])
def upload_target():
    if "image" not in request.files:
        return jsonify({"error": "No image file in request"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    os.makedirs(TARGETS_DIR, exist_ok=True)
    annotations = _read_info()

    idx = len(annotations)
    ext = os.path.splitext(file.filename)[1] or ".jpg"
    crop_name = f"target_{idx:03d}{ext}"
    save_path = os.path.join(TARGETS_DIR, crop_name)

    while os.path.exists(save_path):
        idx += 1
        crop_name = f"target_{idx:03d}{ext}"
        save_path = os.path.join(TARGETS_DIR, crop_name)

    file.save(save_path)

    source = request.form.get("source", "upload")
    annotations.append({
        "source": source,
        "crop": crop_name,
        "bbox": [0, 0, 0, 0],
        "image_size": [0, 0]
    })
    _write_info(annotations)

    if _recognizer is not None:
        _recognizer.reload_targets()

    return jsonify({"message": f"Uploaded {crop_name}", "crop": crop_name}), 201


@app.route("/api/targets/reload", methods=["POST"])
def reload_targets():
    if _recognizer is not None:
        _recognizer.reload_targets()
        return jsonify({"message": "Reload triggered", "count": len(_recognizer.targets)})
    else:
        info_path = os.path.join(TARGETS_DIR, INFO_FILE)
        if os.path.exists(info_path):
            os.utime(info_path, None)
        return jsonify({"message": "Reload signal sent (file mtime updated)"})


@app.route("/api/targets/<name>", methods=["DELETE"])
def delete_target(name):
    annotations = _read_info()

    found = None
    for i, ann in enumerate(annotations):
        if ann["crop"] == name:
            found = i
            break

    if found is None:
        return jsonify({"error": f"Target {name} not found"}), 404

    file_path = os.path.join(TARGETS_DIR, name)
    if os.path.exists(file_path):
        os.remove(file_path)

    annotations.pop(found)
    _write_info(annotations)

    if _recognizer is not None:
        _recognizer.reload_targets()

    return jsonify({"message": f"Deleted {name}", "remaining": len(annotations)})


def run_server(host="0.0.0.0", port=5000, recognizer=None):
    if recognizer is not None:
        set_recognizer(recognizer)
    app.run(host=host, port=port, threaded=True)


if __name__ == "__main__":
    print(f"Target server: http://0.0.0.0:5000")
    print(f"Targets dir: {TARGETS_DIR}")
    run_server()
