"""
目标模板版本管理
- 每次 reload 时自动保存快照
- 支持查看历史和回滚
"""
import os
import json
import shutil
import time

HISTORY_DIR = "target_history"


class TargetHistory:
    def __init__(self, targets_dir="targets", history_dir=HISTORY_DIR):
        self.targets_dir = targets_dir
        self.history_dir = history_dir
        os.makedirs(history_dir, exist_ok=True)

    def save_snapshot(self, label="auto"):
        """保存当前目标集的快照"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        snapshot_dir = os.path.join(self.history_dir, f"{timestamp}_{label}")
        shutil.copytree(self.targets_dir, snapshot_dir)
        # 记录元信息
        meta = {
            "timestamp": timestamp,
            "label": label,
            "files": os.listdir(self.targets_dir)
        }
        with open(os.path.join(snapshot_dir, "_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        return snapshot_dir

    def list_snapshots(self):
        """列出所有快照"""
        snapshots = []
        if not os.path.exists(self.history_dir):
            return snapshots
        for name in sorted(os.listdir(self.history_dir), reverse=True):
            meta_path = os.path.join(self.history_dir, name, "_meta.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                meta["dir_name"] = name
                snapshots.append(meta)
        return snapshots

    def rollback(self, snapshot_name):
        """回滚到指定快照"""
        snapshot_dir = os.path.join(self.history_dir, snapshot_name)
        if not os.path.exists(snapshot_dir):
            raise ValueError(f"快照不存在: {snapshot_name}")
        # 先保存当前状态
        self.save_snapshot(label="before_rollback")
        # 清空当前目标
        for f in os.listdir(self.targets_dir):
            if f.startswith("_"):
                continue
            fpath = os.path.join(self.targets_dir, f)
            if os.path.isfile(fpath):
                os.remove(fpath)
        # 复制快照文件
        for f in os.listdir(snapshot_dir):
            if f.startswith("_"):
                continue
            src = os.path.join(snapshot_dir, f)
            dst = os.path.join(self.targets_dir, f)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
        return True
