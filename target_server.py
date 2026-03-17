"""
目标管理 HTTP API 服务器
端口 5000，提供目标模板的上传、删除、重载、预览、SSE事件流、分组管理、报告导出接口。
敏感接口支持 Bearer Token 认证（设置 TRACKER_API_TOKEN 环境变量启用）。

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
import functools
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



# ==================== API Token 认证 ====================

API_TOKEN = os.environ.get("TRACKER_API_TOKEN", None)


def require_auth(f):
    """装饰器：需要认证的接口"""
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if API_TOKEN is None:
            return f(*args, **kwargs)

        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        if not token:
            token = request.args.get("token", "")

        if token != API_TOKEN:
            return error_response("认证失败", 401)
        return f(*args, **kwargs)
    return decorated


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
let apiToken = localStorage.getItem('tracker_api_token') || '';
let currentGroupFilter = null;

function authHeaders() {
  const h = {};
  if (apiToken) h['Authorization'] = 'Bearer ' + apiToken;
  return h;
}

function setToken() {
  apiToken = document.getElementById('tokenInput').value;
  localStorage.setItem('tracker_api_token', apiToken);
  document.getElementById('authStatus').textContent = apiToken ? '已设置' : '已清除';
}

function loadGroups() {
  fetch('/api/groups').then(r => r.json()).then(data => {
    const tabs = document.getElementById('groupTabs');
    tabs.innerHTML = '';
    const groups = data.groups || {};
    const allCount = Object.values(groups).reduce((a, b) => a + b, 0);
    let tabAll = document.createElement('div');
    tabAll.className = 'group-tab' + (currentGroupFilter === null ? ' active' : '');
    tabAll.innerHTML = '全部<span class="count">(' + allCount + ')</span>';
    tabAll.onclick = () => { currentGroupFilter = null; loadGroups(); loadTargets(); };
    tabs.appendChild(tabAll);
    Object.entries(groups).sort().forEach(([g, cnt]) => {
      const tab = document.createElement('div');
      const label = g || '未分组';
      tab.className = 'group-tab' + (currentGroupFilter === g ? ' active' : '');
      tab.innerHTML = label + '<span class="count">(' + cnt + ')</span>';
      tab.onclick = () => { currentGroupFilter = g; loadGroups(); loadTargets(); };
      tabs.appendChild(tab);
    });
  });
}

function loadTargets() {
  fetch('/api/targets').then(r => r.json()).then(data => {
    const list = document.getElementById('targetList');
    list.innerHTML = '';
    let targets = data.targets;
    if (currentGroupFilter !== null) {
      targets = targets.filter(t => (t.group || '') === currentGroupFilter);
    }
    targets.forEach(t => {
      const card = document.createElement('div');
      card.className = 'target-card';
      const w = t.weight !== undefined ? t.weight : 1.0;
      const mc = t.min_confidence !== undefined ? t.min_confidence : 0.45;
      const g = t.group || '';
      card.innerHTML = `
        <img src="/api/targets/${t.crop}/preview" alt="${t.crop}">
        <div class="name">${t.crop}</div>
        <div class="meta">
          W: <input type="number" value="${w}" step="0.1" min="0.1" class="weight-input"
                    onchange="updateTarget('${t.crop}', this.value, null)">
          MC: <input type="number" value="${mc}" step="0.05" min="0.1" max="1.0" class="weight-input"
                     onchange="updateTarget('${t.crop}', null, this.value)">
        </div>
        <div style="font-size:11px;color:#666;margin:4px 0;">${g ? '分组: ' + g : ''}</div>
        <button class="btn btn-del" onclick="deleteTarget('${t.crop}')">删除</button>
      `;
      list.appendChild(card);
    });
  });
}

function deleteTarget(name) {
  if (!confirm('确定删除 ' + name + '?')) return;
  fetch('/api/targets/' + name, {method: 'DELETE', headers: authHeaders()}).then(r => r.json()).then(d => {
    document.getElementById('status').textContent = d.message || d.error;
    loadTargets(); loadGroups();
  });
}

function reloadTargets() {
  fetch('/api/targets/reload', {method: 'POST', headers: authHeaders()}).then(r => r.json()).then(d => {
    document.getElementById('status').textContent = d.message || d.error;
    loadTargets(); loadGroups();
  });
}

function updateTarget(name, weight, minConf) {
  const body = {};
  if (weight !== null) body.weight = parseFloat(weight);
  if (minConf !== null) body.min_confidence = parseFloat(minConf);
  fetch('/api/targets/' + name, {
    method: 'PUT',
    headers: {...authHeaders(), 'Content-Type': 'application/json'},
    body: JSON.stringify(body)
  }).then(r => r.json()).then(d => {
    document.getElementById('status').textContent = d.message || d.error;
  });
}

document.getElementById('uploadForm').addEventListener('submit', function(e) {
  e.preventDefault();
  const fd = new FormData();
  fd.append('image', document.getElementById('imageInput').files[0]);
  fd.append('weight', document.getElementById('uploadWeight').value);
  fd.append('min_confidence', document.getElementById('uploadMinConf').value);
  fd.append('group', document.getElementById('uploadGroup').value);
  fetch('/api/targets/upload', {method: 'POST', headers: authHeaders(), body: fd}).then(r => r.json()).then(d => {
    document.getElementById('status').textContent = d.message;
    const warnDiv = document.getElementById('similarityWarning');
    if (d.warning) {
      warnDiv.innerHTML = '<div class="similarity-warning">&#9888; ' + d.warning + '</div>';
    } else {
      warnDiv.innerHTML = '';
    }
    if (d.similar_targets && d.similar_targets.length > 0) {
      let html = '<div style="font-size:12px;color:#888;margin:4px 0;">相似目标: ';
      d.similar_targets.forEach(t => { html += t.name + '(' + (t.score * 100).toFixed(0) + '%) '; });
      html += '</div>';
      warnDiv.innerHTML += html;
    }
    loadTargets(); loadGroups();
  });
});

const evtLog = document.getElementById('eventLog');
const evtSource = new EventSource('/api/events');
evtSource.onmessage = function(e) {
  const data = JSON.parse(e.data);
  if (data.type === 'reload' || data.type === 'targets_changed') {
    loadTargets(); loadGroups();
  }
  const div = document.createElement('div');
  div.className = 'event';
  const ts = data.time ? new Date(data.time * 1000).toLocaleTimeString() : '';
  div.textContent = ts + ' [' + data.type + '] ' + JSON.stringify(data);
  evtLog.prepend(div);
  while (evtLog.children.length > 100) evtLog.removeChild(evtLog.lastChild);
};

function updateStats() {
  fetch('/api/stats').then(r => r.json()).then(s => {
    document.getElementById('statFrames').textContent = s.total_frames || 0;
    document.getElementById('statAvgTime').textContent = s.avg_time_ms ? s.avg_time_ms.toFixed(1) : '-';
    document.getElementById('statDetRate').textContent = s.detection_rate ? (s.detection_rate * 100).toFixed(1) + '%' : '-';
    document.getElementById('statTargets').textContent = s.target_count || 0;
    document.getElementById('statThreshold').textContent = s.adaptive_threshold || '-';
  }).catch(() => {});
}
setInterval(updateStats, 5000);
updateStats();

fetch('/api/targets/reload', {method: 'POST'}).then(r => {
  if (r.status === 401) {
    document.getElementById('authBar').style.display = 'flex';
    if (apiToken) document.getElementById('authStatus').textContent = '已保存';
  }
});

loadTargets();
loadGroups();
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
@require_auth
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
@require_auth
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
@require_auth
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
@require_auth
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



# ==================== 分组管理 API ====================

@app.route("/api/groups", methods=["GET"])
def list_groups():
    """列出所有分组和每组目标数"""
    annotations = _read_info()
    groups = {}
    for ann in annotations:
        g = ann.get("group", "")
        groups[g] = groups.get(g, 0) + 1
    active = _recognizer._active_group if _recognizer else None
    return jsonify({"groups": groups, "active_group": active})


@app.route("/api/groups/active", methods=["PUT"])
@require_auth
def set_active_group():
    """设置活跃分组"""
    data = request.get_json(silent=True) or {}
    group = data.get("group", None)
    if group == "" or group is None:
        group = None
    if _recognizer is not None:
        _recognizer.set_active_group(group)
    try:
        from recognize import push_event
        push_event({"type": "group_changed", "group": group or "全部", "time": time.time()})
    except ImportError:
        pass
    return jsonify({"message": f"活跃分组已切换: {group or '全部'}", "active_group": group})


# ==================== 识别报告 ====================

def _get_group_summary():
    """获取分组统计摘要"""
    annotations = _read_info()
    groups = {}
    for ann in annotations:
        g = ann.get("group", "") or "未分组"
        groups[g] = groups.get(g, 0) + 1
    return groups


def _render_report_html(report):
    """生成 HTML 识别报告"""
    summary = report.get("summary", {})
    targets = report.get("targets", [])
    history = report.get("recent_detections", [])
    groups = report.get("groups", {})
    gen_time = report.get("generated_at", "")

    group_items = list(groups.items())
    total_targets = sum(groups.values()) if groups else 1
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22", "#95a5a6"]
    pie_segments = []
    acc = 0
    for i, (g, cnt) in enumerate(group_items):
        pct = cnt / total_targets * 100
        color = colors[i % len(colors)]
        pie_segments.append(f"{color} {acc}% {acc + pct}%")
        acc += pct
    pie_css = f"conic-gradient({', '.join(pie_segments)})" if pie_segments else "#333"

    timeline_html = ""
    if history:
        max_count = max((len(h.get("results", [])) for h in history), default=1) or 1
        for h in history[-30:]:
            ts = h.get("timestamp", 0)
            t_str = time.strftime("%H:%M:%S", time.localtime(ts)) if ts else "?"
            n_det = len(h.get("results", []))
            total_ms = h.get("timing", {}).get("total", 0) * 1000
            bar_w = max(int(n_det / max_count * 200), 2)
            bar_color = "#2ecc71" if n_det > 0 else "#555"
            timeline_html += f'<div style="display:flex;align-items:center;gap:8px;margin:2px 0;">'
            timeline_html += f'<span style="width:70px;font-size:11px;color:#888;">{t_str}</span>'
            timeline_html += f'<div style="width:{bar_w}px;height:14px;background:{bar_color};border-radius:2px;"></div>'
            timeline_html += f'<span style="font-size:11px;color:#aaa;">{n_det}det {total_ms:.0f}ms</span>'
            timeline_html += '</div>'

    target_rows = ""
    for t in targets:
        crop = t.get("crop", "?")
        w = t.get("weight", 1.0)
        mc = t.get("min_confidence", 0.45)
        g = t.get("group", "") or "未分组"
        target_rows += f'<tr><td><img src="/api/targets/{crop}/preview" style="max-width:80px;max-height:60px;border-radius:4px;"></td>'
        target_rows += f'<td>{crop}</td><td>{w}</td><td>{mc}</td><td>{g}</td></tr>'

    legend_html = ""
    for i, (g, cnt) in enumerate(group_items):
        color = colors[i % len(colors)]
        legend_html += f'<span style="display:inline-flex;align-items:center;gap:4px;margin-right:12px;">'
        legend_html += f'<span style="width:12px;height:12px;background:{color};border-radius:2px;display:inline-block;"></span>'
        legend_html += f'{g}: {cnt}</span>'

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>识别报告 - {gen_time}</title>
<style>
body {{font-family:Arial,sans-serif;max-width:960px;margin:0 auto;padding:20px;background:#1a1a2e;color:#eee;}}
h1 {{color:#0ff;}} h2 {{color:#0ff;margin-top:30px;border-bottom:1px solid #333;padding-bottom:8px;}}
.stats-grid {{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:16px 0;}}
.stat-card {{background:#16213e;padding:16px;border-radius:8px;text-align:center;}}
.stat-card .value {{font-size:28px;color:#0ff;font-weight:bold;}}
.stat-card .label {{font-size:12px;color:#888;margin-top:4px;}}
table {{width:100%;border-collapse:collapse;margin:12px 0;}}
th {{background:#16213e;padding:8px;text-align:left;color:#0ff;}}
td {{padding:8px;border-bottom:1px solid #222;}}
.pie {{width:120px;height:120px;border-radius:50%;background:{pie_css};margin:12px auto;}}
.timeline {{background:#0d1117;padding:12px;border-radius:8px;max-height:400px;overflow-y:auto;}}
.footer {{margin-top:30px;text-align:center;color:#555;font-size:11px;}}
</style></head><body>
<h1>识别报告</h1><p style="color:#888;">生成时间: {gen_time}</p>
<h2>概要统计</h2><div class="stats-grid">
<div class="stat-card"><div class="value">{summary.get('total_frames',0)}</div><div class="label">处理帧数</div></div>
<div class="stat-card"><div class="value">{summary.get('avg_time_ms',0):.1f}ms</div><div class="label">平均耗时</div></div>
<div class="stat-card"><div class="value">{summary.get('detection_rate',0)*100:.1f}%</div><div class="label">检测率</div></div>
<div class="stat-card"><div class="value">{len(targets)}</div><div class="label">目标数量</div></div></div>
<h2>分组统计</h2><div style="display:flex;align-items:center;gap:30px;">
<div class="pie"></div><div>{legend_html}</div></div>
<h2>目标列表</h2><table>
<tr><th>缩略图</th><th>名称</th><th>权重</th><th>最低置信度</th><th>分组</th></tr>
{target_rows}</table>
<h2>最近检测时间线</h2><div class="timeline">
{timeline_html if timeline_html else '<p style="color:#555;">暂无检测数据</p>'}
</div><div class="footer">A8mini Target Tracker v1.5 - 自动生成报告</div></body></html>"""
    return html


@app.route("/api/report", methods=["GET"])
def generate_report():
    """生成识别统计报告"""
    fmt = request.args.get("format", "json")
    stats = {}
    history = []
    try:
        from recognize import recognition_history
        stats = recognition_history.stats()
        history = recognition_history.recent(n=50)
    except ImportError:
        pass
    targets = _read_info()
    groups = _get_group_summary()
    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": stats,
        "targets": targets,
        "recent_detections": history,
        "groups": groups,
    }
    if fmt == "html":
        html = _render_report_html(report)
        return Response(html, mimetype="text/html")
    else:
        return jsonify(report)


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
