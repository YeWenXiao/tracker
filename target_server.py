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
  GET    /api/targets/export         - 导出目标为 ZIP
  POST   /api/targets/import         - 从 ZIP 导入目标
  GET    /api/history                - 列出目标模板历史快照
  POST   /api/history/rollback       - 回滚到指定快照
  GET    /api/history/detections     - 最近N次识别结果
  GET    /api/stats                  - 识别统计信息
  GET    /api/events                 - SSE 实时事件流
"""

import os
import json
import time
import threading
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file, Response

from target_history import TargetHistory

app = Flask(__name__)

TARGETS_DIR = os.environ.get("TARGETS_DIR", "targets")
INFO_FILE = "target_info.json"

_recognizer = None
_target_history = TargetHistory()


# ==================== 统一错误处理 ====================

def error_response(message, status_code, details=None):
    """统一错误响应格式"""
    resp = {"error": message, "status": status_code}
    if details:
        resp["details"] = details
    return jsonify(resp), status_code


@app.errorhandler(400)
def bad_request(e):
    return error_response("请求参数错误", 400)


@app.errorhandler(404)
def not_found(e):
    return error_response("资源未找到", 404)


@app.errorhandler(500)
def internal_error(e):
    return error_response("服务器内部错误", 500, str(e))


# ==================== 文件校验 ====================

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


def validate_image(file):
    """校验上传的图片文件"""
    if not file or file.filename == "":
        return "未选择文件"
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return f"不支持的文件类型: {ext}，支持: {ALLOWED_EXTENSIONS}"
    file.seek(0, 2)
    size = file.tell()
    file.seek(0)
    if size > MAX_FILE_SIZE:
        return f"文件过大: {size/1024/1024:.1f}MB，最大: {MAX_FILE_SIZE/1024/1024:.0f}MB"
    return None


def set_recognizer(rec):
    """设置外部 recognizer 实例，reload API 会直接调用其 reload_targets()"""
    global _recognizer
    _recognizer = rec


# ==================== 并发安全的文件读写 ====================

_file_lock = threading.Lock()


def _read_info():
    """读取 target_info.json（线程安全）"""
    with _file_lock:
        info_path = os.path.join(TARGETS_DIR, INFO_FILE)
        if not os.path.exists(info_path):
            return []
        with open(info_path, "r", encoding="utf-8") as f:
            return json.load(f)


def _write_info(annotations):
    """写入 target_info.json（线程安全 + 原子写入）"""
    with _file_lock:
        info_path = os.path.join(TARGETS_DIR, INFO_FILE)
        tmp_path = info_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(annotations, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, info_path)


# ==================== HTML 管理页面 ====================

@app.route("/", methods=["GET"])
def index():
    """目标管理系统页面"""
    html = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>目标管理系统</title>
<style>
  body { font-family: Arial, sans-serif; max-width: 960px; margin: 0 auto; padding: 20px; background: #1a1a2e; color: #eee; }
  h1 { color: #0ff; }
  h3 { color: #ccc; margin-top: 24px; }
  .targets { display: flex; flex-wrap: wrap; gap: 16px; }
  .target-card { background: #16213e; border-radius: 8px; padding: 12px; width: 200px; text-align: center; }
  .target-card img { max-width: 180px; max-height: 140px; border-radius: 4px; }
  .target-card .name { font-size: 12px; color: #aaa; margin: 6px 0; word-break: break-all; }
  .target-card .meta { font-size: 11px; color: #888; margin: 4px 0; }
  .target-card .meta-label { color: #0ff; font-size: 10px; }
  .btn { padding: 6px 12px; border: none; border-radius: 4px; cursor: pointer; font-size: 13px; }
  .btn-del { background: #e74c3c; color: #fff; margin-top: 6px; }
  .btn-del:hover { background: #c0392b; }
  .btn-reload { background: #2ecc71; color: #fff; margin: 10px 4px; }
  .btn-reload:hover { background: #27ae60; }
  .btn-rollback { background: #9b59b6; color: #fff; margin: 2px; font-size: 11px; padding: 4px 8px; }
  .btn-rollback:hover { background: #8e44ad; }
  .upload-area { background: #16213e; padding: 20px; border-radius: 8px; margin: 20px 0; }
  .upload-area input[type=file] { margin: 8px 0; }
  .btn-upload { background: #3498db; color: #fff; }
  .btn-upload:hover { background: #2980b9; }
  .events { background: #0d1117; padding: 10px; border-radius: 4px; max-height: 200px; overflow-y: auto;
            font-family: monospace; font-size: 12px; margin-top: 10px; }
  .events .event { padding: 2px 0; color: #0f0; }
  .weight-input { width: 50px; background: #222; color: #fff; border: 1px solid #444; border-radius: 3px; text-align: center; }
  .stats-panel { background: #16213e; padding: 16px; border-radius: 8px; margin: 20px 0; display: grid;
                 grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }
  .stat-item { text-align: center; }
  .stat-item .value { font-size: 24px; color: #0ff; font-weight: bold; }
  .stat-item .label { font-size: 12px; color: #888; margin-top: 4px; }
  .similarity-warning { background: #e74c3c33; border: 1px solid #e74c3c; padding: 8px 12px;
                        border-radius: 4px; margin: 8px 0; color: #e74c3c; }
  .rollback-panel { background: #16213e; padding: 12px; border-radius: 8px; margin: 10px 0; }
  .snapshot { display: flex; justify-content: space-between; align-items: center;
              padding: 6px 0; border-bottom: 1px solid #333; font-size: 12px; }
  .snapshot:last-child { border-bottom: none; }
  .snap-info { color: #aaa; }
  .snap-files { color: #666; font-size: 11px; }
</style>
</head>
<body>
<h1>目标管理系统</h1>

<div class="upload-area">
  <h3 style="margin-top:0">上传目标</h3>
  <form id="uploadForm" enctype="multipart/form-data">
    <input type="file" id="imageInput" name="image" accept="image/*">
    <label>权重: <input type="number" id="uploadWeight" value="1.0" step="0.1" min="0.1" class="weight-input"></label>
    <label>最低置信度: <input type="number" id="uploadMinConf" value="0.45" step="0.05" min="0.1" max="1.0" class="weight-input"></label>
    <button type="submit" class="btn btn-upload">上传目标</button>
  </form>
</div>

<h3>识别统计</h3>
<div class="stats-panel" id="statsPanel">
  <div class="stat-item"><div class="value" id="statFrames">-</div><div class="label">总帧数</div></div>
  <div class="stat-item"><div class="value" id="statAvgTime">-</div><div class="label">平均耗时(ms)</div></div>
  <div class="stat-item"><div class="value" id="statDetRate">-</div><div class="label">检测率</div></div>
  <div class="stat-item"><div class="value" id="statTargets">-</div><div class="label">目标数量</div></div>
  <div class="stat-item"><div class="value" id="statThreshold">-</div><div class="label">自适应阈值</div></div>
</div>

<button class="btn btn-reload" onclick="reloadTargets()">刷新</button>
<span id="status" style="color:#0f0;"></span>
<div id="similarityWarning"></div>

<h3>目标列表</h3>
<div class="targets" id="targetList"></div>

<h3>历史快照</h3>
<div class="rollback-panel" id="rollbackPanel">
  <div style="color:#888;font-size:12px;">加载中...</div>
</div>

<h3>事件日志</h3>
<div class="events" id="eventLog"></div>

<script>
function loadTargets() {
  fetch("/api/targets").then(function(r){return r.json()}).then(function(data){
    var list = document.getElementById("targetList");
    list.innerHTML = "";
    data.targets.forEach(function(t){
      var card = document.createElement("div");
      card.className = "target-card";
      var w = t.weight !== undefined ? t.weight : 1.0;
      var mc = t.min_confidence !== undefined ? t.min_confidence : 0.45;
      card.innerHTML =
        '<img src="/api/targets/' + t.crop + '/preview" alt="' + t.crop + '">' +
        '<div class="name">' + t.crop + '</div>' +
        '<div class="meta">' +
        '<span class="meta-label">权重</span> ' +
        '<input type="number" value="' + w + '" step="0.1" min="0.1" class="weight-input" ' +
            "onchange=\"updateTarget('" + t.crop + "', this.value, null)\"> " +
        '<span class="meta-label">阈值</span> ' +
        '<input type="number" value="' + mc + '" step="0.05" min="0.1" max="1.0" class="weight-input" ' +
            "onchange=\"updateTarget('" + t.crop + "', null, this.value)\">" +
        '</div>' +
        '<button class="btn btn-del" onclick="deleteTarget(\'' + t.crop + '\')">删除</button>';
      list.appendChild(card);
    });
  });
}

function deleteTarget(name) {
  if (!confirm("确定删除目标 " + name + " ?")) return;
  fetch("/api/targets/" + name, {method:"DELETE"}).then(function(r){return r.json()}).then(function(d){
    document.getElementById("status").textContent = d.message || d.error;
    loadTargets(); loadSnapshots();
  });
}

function reloadTargets() {
  fetch("/api/targets/reload", {method:"POST"}).then(function(r){return r.json()}).then(function(d){
    document.getElementById("status").textContent = d.message || d.error;
    loadTargets();
  });
}

function updateTarget(name, weight, minConf) {
  var body = {};
  if (weight !== null) body.weight = parseFloat(weight);
  if (minConf !== null) body.min_confidence = parseFloat(minConf);
  fetch("/api/targets/" + name, {
    method: "PUT",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body)
  }).then(function(r){return r.json()}).then(function(d){
    document.getElementById("status").textContent = d.message || d.error;
  });
}

document.getElementById("uploadForm").addEventListener("submit", function(e) {
  e.preventDefault();
  var fd = new FormData();
  var fi = document.getElementById("imageInput");
  if (!fi.files[0]) { alert("请先选择图片文件"); return; }
  fd.append("image", fi.files[0]);
  fd.append("weight", document.getElementById("uploadWeight").value);
  fd.append("min_confidence", document.getElementById("uploadMinConf").value);
  fetch("/api/targets/upload", {method:"POST", body:fd}).then(function(r){return r.json()}).then(function(d){
    document.getElementById("status").textContent = d.message || d.error;
    var warnDiv = document.getElementById("similarityWarning");
    if (d.warning) {
      warnDiv.innerHTML = '<div class="similarity-warning">&#9888; 注意: ' + d.warning + '</div>';
    } else {
      warnDiv.innerHTML = "";
    }
    if (d.similar_targets && d.similar_targets.length > 0) {
      var html = '<div style="font-size:12px;color:#888;margin:4px 0;">相似目标: ';
      d.similar_targets.forEach(function(t){ html += t.name + "(" + (t.score*100).toFixed(0) + "%) "; });
      html += "</div>";
      warnDiv.innerHTML += html;
    }
    loadTargets(); loadSnapshots();
  });
});

function loadSnapshots() {
  fetch("/api/history").then(function(r){return r.json()}).then(function(data){
    var panel = document.getElementById("rollbackPanel");
    var snaps = data.snapshots || [];
    if (snaps.length === 0) {
      panel.innerHTML = '<div style="color:#888;font-size:12px;">暂无历史快照</div>';
      return;
    }
    var html = "";
    snaps.slice(0, 3).forEach(function(s){
      var fc = s.files ? s.files.length : 0;
      html += '<div class="snapshot">' +
        '<span class="snap-info">' + s.timestamp + ' [' + s.label + '] ' +
        '<span class="snap-files">' + fc + ' 个文件</span></span>' +
        "<button class=\"btn btn-rollback\" onclick=\"rollbackTo('" + s.dir_name + "')\">回滚</button>" +
        '</div>';
    });
    panel.innerHTML = html;
  }).catch(function(){
    document.getElementById("rollbackPanel").innerHTML =
      '<div style="color:#888;font-size:12px;">无法加载快照</div>';
  });
}

function rollbackTo(snapName) {
  if (!confirm("确定回滚到快照 " + snapName + " ?\n当前目标将被替换。")) return;
  fetch("/api/history/rollback", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({snapshot: snapName})
  }).then(function(r){return r.json()}).then(function(d){
    document.getElementById("status").textContent = d.message || d.error;
    loadTargets(); loadSnapshots();
  });
}

var evtLog = document.getElementById("eventLog");
var evtSource = new EventSource("/api/events");
evtSource.onmessage = function(e) {
  var data = JSON.parse(e.data);
  if (data.type === "reload" || data.type === "targets_changed") {
    loadTargets(); loadSnapshots();
  }
  var div = document.createElement("div");
  div.className = "event";
  var ts = data.time ? new Date(data.time * 1000).toLocaleTimeString() : "";
  div.textContent = ts + " [" + data.type + "] " + JSON.stringify(data);
  evtLog.prepend(div);
  while (evtLog.children.length > 100) evtLog.removeChild(evtLog.lastChild);
};

function updateStats() {
  fetch("/api/stats").then(function(r){return r.json()}).then(function(s){
    document.getElementById("statFrames").textContent = s.total_frames || 0;
    document.getElementById("statAvgTime").textContent = s.avg_time_ms ? s.avg_time_ms.toFixed(1) : "-";
    document.getElementById("statDetRate").textContent = s.detection_rate ? (s.detection_rate*100).toFixed(1)+"%" : "-";
    document.getElementById("statTargets").textContent = s.target_count || 0;
    document.getElementById("statThreshold").textContent = s.adaptive_threshold || "-";
  }).catch(function(){});
}
setInterval(updateStats, 5000);
updateStats();
loadTargets();
loadSnapshots();
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
        return error_response("请求中未包含图片文件", 400)

    file = request.files["image"]
    err = validate_image(file)
    if err:
        return error_response(err, 400)

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

    # 检查与已有目标的相似度
    warning = None
    similar_targets = []
    if _recognizer:
        img = cv2.imread(save_path)
        if img is not None:
            sims = _recognizer.check_similarity(img)
            similar_targets = [{"name": n, "score": round(s, 4)} for n, s in sims[:3]]
            if sims and sims[0][1] > 0.8:
                warning = f"与 {sims[0][0]} 相似度 {sims[0][1]:.2f}"

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

    # 推送 SSE targets_changed 事件
    try:
        from recognize import push_event
        push_event({"type": "targets_changed", "action": "upload", "target": crop_name, "time": time.time()})
    except ImportError:
        pass

    result = {"message": f"已上传 {crop_name}", "crop": crop_name}
    if warning:
        result["warning"] = warning
    if similar_targets:
        result["similar_targets"] = similar_targets
    return jsonify(result), 201




@app.route("/api/targets/check-similarity", methods=["POST"])
def check_similarity():
    """上传图片检查与已有目标的相似度（不保存）"""
    if "image" not in request.files:
        return error_response("请求中未包含图片文件", 400)

    if _recognizer is None:
        return error_response("识别引擎未初始化", 503)

    file = request.files["image"]
    err = validate_image(file)
    if err:
        return error_response(err, 400)

    # 读取图片到内存
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return error_response("无法解析图片", 400)

    sims = _recognizer.check_similarity(img)
    results = [{"name": n, "score": round(s, 4)} for n, s in sims]
    is_duplicate = bool(sims and sims[0][1] > 0.8)

    return jsonify({
        "similarities": results,
        "is_duplicate": is_duplicate,
        "threshold": 0.8,
    })

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
        return error_response(f"目标 {name} 未找到", 404)

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
        return error_response(f"目标 {name} 未找到", 404)

    file_path = os.path.join(TARGETS_DIR, name)
    if os.path.exists(file_path):
        os.remove(file_path)

    annotations.pop(found)
    _write_info(annotations)

    if _recognizer is not None:
        _recognizer.reload_targets()

    # 推送 SSE targets_changed 事件
    try:
        from recognize import push_event
        push_event({"type": "targets_changed", "action": "delete", "target": name, "time": time.time()})
    except ImportError:
        pass

    return jsonify({"message": f"已删除 {name}", "remaining": len(annotations)})




@app.route("/api/stats", methods=["GET"])
def get_stats():
    """获取识别统计数据"""
    try:
        from recognize import recognition_history
        stats = recognition_history.stats()
        stats["target_count"] = len(_recognizer.targets) if _recognizer else 0
        if _recognizer:
            stats["adaptive_threshold"] = round(_recognizer.adaptive_threshold.get(), 4)
        return jsonify(stats)
    except ImportError:
        return error_response("recognize 模块不可用", 503)

@app.route("/api/targets/<name>/preview", methods=["GET"])
def preview_target(name):
    """返回目标模板的缩略图"""
    file_path = os.path.join(TARGETS_DIR, name)
    if not os.path.exists(file_path):
        return error_response(f"目标 {name} 的模板文件未找到", 404)
    return send_file(file_path, mimetype="image/jpeg")




# ==================== 历史管理 API ====================

@app.route("/api/history", methods=["GET"])
def list_history():
    """列出目标模板的历史快照"""
    snapshots = _target_history.list_snapshots()
    return jsonify({"snapshots": snapshots, "count": len(snapshots)})


@app.route("/api/history/rollback", methods=["POST"])
def rollback_history():
    """回滚到指定快照"""
    data = request.get_json(silent=True) or {}
    snapshot = data.get("snapshot")
    if not snapshot:
        return error_response("需要 snapshot 参数", 400)
    try:
        _target_history.rollback(snapshot)
        # 触发 reload
        if _recognizer is not None:
            _recognizer.reload_targets()
        return jsonify({"message": f"已回滚到 {snapshot}"})
    except ValueError as e:
        return error_response(str(e), 404)


@app.route("/api/history/detections", methods=["GET"])
def detection_history():
    """返回最近N次识别结果"""
    try:
        from recognize import recognition_history
    except ImportError:
        return error_response("需要 recognize 模块", 500)
    n = request.args.get("n", 10, type=int)
    recent = recognition_history.recent(n)
    return jsonify({"detections": recent, "count": len(recent)})


# ==================== 批量导入/导出 ====================

@app.route("/api/targets/export", methods=["GET"])
def export_targets():
    """打包当前目标为 ZIP 下载"""
    import io
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in os.listdir(TARGETS_DIR):
            filepath = os.path.join(TARGETS_DIR, f)
            if os.path.isfile(filepath):
                zf.write(filepath, f)
    buf.seek(0)
    return send_file(buf, mimetype="application/zip",
                     as_attachment=True, download_name="targets.zip")


@app.route("/api/targets/import", methods=["POST"])
def import_targets():
    """从 ZIP 导入目标（替换当前所有目标）"""
    if "file" not in request.files:
        return error_response("需要 file 字段", 400)
    import zipfile
    import io
    file = request.files["file"]
    if not file.filename.endswith(".zip"):
        return error_response("需要 ZIP 文件", 400)
    # 保存当前状态快照
    try:
        _target_history.save_snapshot(label="before_import")
    except Exception:
        pass
    # 清空当前目标目录
    for f in os.listdir(TARGETS_DIR):
        fpath = os.path.join(TARGETS_DIR, f)
        if os.path.isfile(fpath):
            os.remove(fpath)
    # 解压到 targets/ 目录
    with zipfile.ZipFile(io.BytesIO(file.read())) as zf:
        zf.extractall(TARGETS_DIR)
    # 触发 reload
    if _recognizer is not None:
        _recognizer.reload_targets()
    annotations = _read_info()
    return jsonify({"message": "导入成功", "count": len(annotations)})


# ==================== SSE 事件流 ====================

@app.route("/api/events", methods=["GET"])
def events():
    """SSE 事件流，推送识别结果和重载通知"""
    try:
        from recognize import register_sse_client, unregister_sse_client
    except ImportError:
        return error_response("SSE 需要 recognize 模块", 500)

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
