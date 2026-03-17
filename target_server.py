"""
目标管理 HTTP API 服务器
端口 5000，提供目标模板的上传、删除、重载、预览、SSE事件流接口。

可独立运行:
  python target_server.py

也可作为 recognize.py 的子线程启动。

API:
  GET    /                           - 目标管理页面（HTML）
  GET    /api/targets                - 获取当前目标列表
  POST   /api/targets/upload         - 上传新目标图片 (multipart/form-data, field="image")
  POST   /api/targets/reload         - 触发重新加载
  PUT    /api/targets/<name>         - 更新目标的 weight / min_confidence
  DELETE /api/targets/<name>         - 删除某个目标
  GET    /api/targets/<name>/preview - 获取目标模板缩略图
  GET    /api/events                 - SSE 实时事件流
"""

import os
import json
import time
from flask import Flask, request, jsonify, send_file, Response

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


# ==================== HTML 管理页面 ====================

@app.route("/", methods=["GET"])
def index():
    """简单的目标管理页面"""
    html = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>目标管理</title>
<style>
  body { font-family: Arial, sans-serif; max-width: 960px; margin: 0 auto; padding: 20px; background: #1a1a2e; color: #eee; }
  h1 { color: #0ff; }
  .targets { display: flex; flex-wrap: wrap; gap: 16px; }
  .target-card { background: #16213e; border-radius: 8px; padding: 12px; width: 200px; text-align: center; }
  .target-card img { max-width: 180px; max-height: 140px; border-radius: 4px; }
  .target-card .name { font-size: 12px; color: #aaa; margin: 6px 0; word-break: break-all; }
  .target-card .meta { font-size: 11px; color: #888; }
  .btn { padding: 6px 12px; border: none; border-radius: 4px; cursor: pointer; font-size: 13px; }
  .btn-del { background: #e74c3c; color: #fff; }
  .btn-del:hover { background: #c0392b; }
  .btn-reload { background: #2ecc71; color: #fff; margin: 10px 0; }
  .btn-reload:hover { background: #27ae60; }
  .upload-area { background: #16213e; padding: 20px; border-radius: 8px; margin: 20px 0; }
  .upload-area input[type=file] { margin: 8px 0; }
  .btn-upload { background: #3498db; color: #fff; }
  .btn-upload:hover { background: #2980b9; }
  .events { background: #0d1117; padding: 10px; border-radius: 4px; max-height: 200px; overflow-y: auto;
            font-family: monospace; font-size: 12px; margin-top: 20px; }
  .events .event { padding: 2px 0; color: #0f0; }
  .weight-input { width: 50px; background: #222; color: #fff; border: 1px solid #444; border-radius: 3px; text-align: center; }
</style>
</head>
<body>
<h1>A8mini 目标管理</h1>

<div class="upload-area">
  <h3>上传新目标</h3>
  <form id="uploadForm" enctype="multipart/form-data">
    <input type="file" id="imageInput" name="image" accept="image/*">
    <label>权重: <input type="number" id="uploadWeight" value="1.0" step="0.1" min="0.1" class="weight-input"></label>
    <label>最低置信度: <input type="number" id="uploadMinConf" value="0.45" step="0.05" min="0.1" max="1.0" class="weight-input"></label>
    <button type="submit" class="btn btn-upload">上传</button>
  </form>
</div>

<button class="btn btn-reload" onclick="reloadTargets()">重新加载模板</button>
<span id="status"></span>

<div class="targets" id="targetList"></div>

<h3>实时事件</h3>
<div class="events" id="eventLog"></div>

<script>
function loadTargets() {
  fetch('/api/targets').then(r => r.json()).then(data => {
    const list = document.getElementById('targetList');
    list.innerHTML = '';
    data.targets.forEach(t => {
      const card = document.createElement('div');
      card.className = 'target-card';
      const w = t.weight !== undefined ? t.weight : 1.0;
      const mc = t.min_confidence !== undefined ? t.min_confidence : 0.45;
      card.innerHTML =
        '<img src="/api/targets/' + t.crop + '/preview" alt="' + t.crop + '">' +
        '<div class="name">' + t.crop + '</div>' +
        '<div class="meta">' +
        'W: <input type="number" value="' + w + '" step="0.1" min="0.1" class="weight-input" ' +
            'onchange="updateTarget(\\'' + t.crop + '\\', this.value, null)"> ' +
        'MC: <input type="number" value="' + mc + '" step="0.05" min="0.1" max="1.0" class="weight-input" ' +
            'onchange="updateTarget(\\'' + t.crop + '\\', null, this.value)">' +
        '</div>' +
        '<button class="btn btn-del" onclick="deleteTarget(\\'' + t.crop + '\\')">删除</button>';
      list.appendChild(card);
    });
  });
}

function deleteTarget(name) {
  if (!confirm('确定删除 ' + name + '?')) return;
  fetch('/api/targets/' + name, {method: 'DELETE'}).then(r => r.json()).then(d => {
    document.getElementById('status').textContent = d.message;
    loadTargets();
  });
}

function reloadTargets() {
  fetch('/api/targets/reload', {method: 'POST'}).then(r => r.json()).then(d => {
    document.getElementById('status').textContent = d.message;
    loadTargets();
  });
}

function updateTarget(name, weight, minConf) {
  var body = {};
  if (weight !== null) body.weight = parseFloat(weight);
  if (minConf !== null) body.min_confidence = parseFloat(minConf);
  fetch('/api/targets/' + name, {
    method: 'PUT',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body)
  }).then(r => r.json()).then(d => {
    document.getElementById('status').textContent = d.message || d.error;
  });
}

document.getElementById('uploadForm').addEventListener('submit', function(e) {
  e.preventDefault();
  var fd = new FormData();
  fd.append('image', document.getElementById('imageInput').files[0]);
  fd.append('weight', document.getElementById('uploadWeight').value);
  fd.append('min_confidence', document.getElementById('uploadMinConf').value);
  fetch('/api/targets/upload', {method: 'POST', body: fd}).then(r => r.json()).then(d => {
    document.getElementById('status').textContent = d.message;
    loadTargets();
  });
});

// SSE 事件流
var evtLog = document.getElementById('eventLog');
var evtSource = new EventSource('/api/events');
evtSource.onmessage = function(e) {
  var div = document.createElement('div');
  div.className = 'event';
  div.textContent = e.data;
  evtLog.prepend(div);
  while (evtLog.children.length > 100) evtLog.removeChild(evtLog.lastChild);
};

loadTargets();
</script>
</body>
</html>"""
    return html


# ==================== REST API ====================

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

    # 读取可选的 weight 和 min_confidence 参数
    weight = float(request.form.get("weight", 1.0))
    min_confidence = float(request.form.get("min_confidence", 0.45))

    source = request.form.get("source", "upload")
    annotations.append({
        "source": source,
        "crop": crop_name,
        "bbox": [0, 0, 0, 0],
        "image_size": [0, 0],
        "weight": weight,
        "min_confidence": min_confidence,
    })
    _write_info(annotations)

    if _recognizer is not None:
        _recognizer.reload_targets()

    return jsonify({"message": f"已上传 {crop_name}", "crop": crop_name}), 201


@app.route("/api/targets/reload", methods=["POST"])
def reload_targets():
    if _recognizer is not None:
        _recognizer.reload_targets()
        return jsonify({"message": "已触发重载", "count": len(_recognizer.targets)})
    else:
        info_path = os.path.join(TARGETS_DIR, INFO_FILE)
        if os.path.exists(info_path):
            os.utime(info_path, None)
        return jsonify({"message": "已发送重载信号 (更新文件时间戳)"})


@app.route("/api/targets/<name>", methods=["PUT"])
def update_target(name):
    """更新目标的 weight 和 min_confidence"""
    annotations = _read_info()

    found = None
    for i, ann in enumerate(annotations):
        if ann["crop"] == name:
            found = i
            break

    if found is None:
        return jsonify({"error": f"目标 {name} 未找到"}), 404

    data = request.get_json(silent=True) or {}
    if "weight" in data:
        annotations[found]["weight"] = float(data["weight"])
    if "min_confidence" in data:
        annotations[found]["min_confidence"] = float(data["min_confidence"])

    _write_info(annotations)

    if _recognizer is not None:
        _recognizer.reload_targets()

    return jsonify({
        "message": f"已更新 {name}",
        "target": annotations[found],
    })


@app.route("/api/targets/<name>", methods=["DELETE"])
def delete_target(name):
    annotations = _read_info()

    found = None
    for i, ann in enumerate(annotations):
        if ann["crop"] == name:
            found = i
            break

    if found is None:
        return jsonify({"error": f"目标 {name} 未找到"}), 404

    file_path = os.path.join(TARGETS_DIR, name)
    if os.path.exists(file_path):
        os.remove(file_path)

    annotations.pop(found)
    _write_info(annotations)

    if _recognizer is not None:
        _recognizer.reload_targets()

    return jsonify({"message": f"已删除 {name}", "remaining": len(annotations)})


@app.route("/api/targets/<name>/preview", methods=["GET"])
def preview_target(name):
    """返回目标模板的缩略图"""
    file_path = os.path.join(TARGETS_DIR, name)
    if not os.path.exists(file_path):
        return jsonify({"error": "未找到"}), 404
    return send_file(file_path, mimetype="image/jpeg")


# ==================== SSE 事件流 ====================

@app.route("/api/events", methods=["GET"])
def events():
    """SSE 事件流，推送识别结果和重载通知"""
    try:
        from recognize import register_sse_client, unregister_sse_client
    except ImportError:
        return jsonify({"error": "SSE 需要 recognize 模块"}), 500

    def generate():
        q = register_sse_client()
        try:
            while True:
                try:
                    event = q.get(timeout=30)
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                except Exception:
                    # 超时时发送心跳保持连接
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
        finally:
            unregister_sse_client(q)

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def run_server(host="0.0.0.0", port=5000, recognizer=None):
    """启动服务器（可从外部调用）"""
    if recognizer is not None:
        set_recognizer(recognizer)
    app.run(host=host, port=port, threaded=True)


if __name__ == "__main__":
    print(f"目标管理服务器启动: http://0.0.0.0:5000")
    print(f"目标目录: {TARGETS_DIR}")
    run_server()
