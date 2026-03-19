"""
Web MJPEG 监控服务器 — 新增特征匹配状态显示
"""

import json
import time
import threading
import cv2
from http.server import HTTPServer, BaseHTTPRequestHandler
from config import WEB_PORT


HTML_PAGE = '''<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>V2.0 Tracker</title>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { background:#111; color:#0f0; font-family:monospace; display:flex; flex-direction:column; align-items:center; }
h1 { font-size:16px; padding:8px; }
#container { position:relative; display:inline-block; }
#video { border:2px solid #0f0; max-width:95vw; max-height:80vh; }
#status { position:absolute; top:10px; left:10px; background:rgba(0,0,0,0.7); color:#0f0; padding:6px 12px; font-size:14px; border-radius:4px; white-space:pre-line; }
#btns { padding:5px; }
#btns button { background:#333; color:#0f0; border:1px solid #0f0; padding:6px 16px; margin:0 5px; cursor:pointer; font-family:monospace; }
#btns button:hover { background:#0f0; color:#111; }
</style>
</head><body>
<h1>V2.0 Target Tracker</h1>
<div id="btns">
    <button onclick="doAction('center')">云台回中</button>
    <button onclick="doAction('rescan')">重新搜索</button>
    <button onclick="doAction('zoom_in')">Zoom+</button>
    <button onclick="doAction('zoom_out')">Zoom-</button>
</div>
<div id="container">
    <img id="video" src="/stream">
    <div id="status">启动中...</div>
</div>
<script>
function doAction(act) {
    fetch('/' + act).then(r => r.json()).then(d => {
        document.getElementById('status').textContent = d.msg || 'OK';
    });
}
setInterval(function() {
    fetch('/status').then(r => r.json()).then(d => {
        document.getElementById('status').textContent = d.text;
    }).catch(() => {});
}, 500);
</script>
</body></html>'''


class WebServer:
    def __init__(self, port=None):
        self.port = port or WEB_PORT
        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self.status_text = '启动中...'
        self.callbacks = {}

        server_ref = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                path = self.path.split('?')[0]

                if path == '/':
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html; charset=utf-8')
                    self.end_headers()
                    self.wfile.write(HTML_PAGE.encode('utf-8'))

                elif path == '/stream':
                    self.send_response(200)
                    self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                    self.end_headers()
                    try:
                        while server_ref.running:
                            with server_ref.lock:
                                f = server_ref.frame
                            if f is not None:
                                small = cv2.resize(f, (640, 360))
                                ret, jpg = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 60])
                                if ret:
                                    data = jpg.tobytes()
                                    self.wfile.write(b'--frame\r\n')
                                    self.wfile.write(b'Content-Type: image/jpeg\r\n')
                                    self.wfile.write(f'Content-Length: {len(data)}\r\n'.encode())
                                    self.wfile.write(b'\r\n')
                                    self.wfile.write(data)
                                    self.wfile.write(b'\r\n')
                            time.sleep(0.08)
                    except (BrokenPipeError, ConnectionResetError):
                        pass

                elif path == '/status':
                    self._json({'text': server_ref.status_text})

                elif path.lstrip('/') in server_ref.callbacks:
                    action = path.lstrip('/')
                    msg = server_ref.callbacks[action]()
                    self._json({'ok': True, 'msg': msg})

                else:
                    self.send_response(404)
                    self.end_headers()

            def _json(self, data):
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(data).encode())

            def log_message(self, format, *args):
                pass

        self.server = HTTPServer(('0.0.0.0', self.port), Handler)

    def start(self):
        self.running = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def update_frame(self, frame):
        with self.lock:
            self.frame = frame

    def stop(self):
        self.running = False
        self.server.shutdown()
