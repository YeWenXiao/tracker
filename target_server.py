"""
目标管理 HTTP API 服务器
端口 5000，提供目标模板的上传、删除、重载接口。

可独立运行:
  python target_server.py

也可作为 recognize.py 的子线程启动。

API:
  POST   /api/targets/upload  - 上传新目标图片 (multipart/form-data, field="image")
  POST   /api/targets/reload  - 触发重新加载
  GET    /api/targets         - 获取当前目标列表
  DELETE /api/targets/<name>  - 删除某个目标
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
    """设置外部 recognizer 实例，reload API 会直接调用其 reload_targets()"""
    global _recognizer
    _recognizer = rec


def _read_info():
    """读取 target_info.json"""
    info_path = os.path.join(TARGETS_DIR, INFO_FILE)
    if not os.path.exists(info_path):
        return []
    with open(info_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_info(annotations):
    """写入 target_info.json"""
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
    """启动服务器（可从外部调用）"""
    if recognizer is not None:
        set_recognizer(recognizer)
    app.run(host=host, port=port, threaded=True)


if __name__ == "__main__":
    print(f"目标管理服务器启动: http://0.0.0.0:5000")
    print(f"目标目录: {TARGETS_DIR}")
    run_server()
