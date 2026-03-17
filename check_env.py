"""
Jetson 环境检测脚本
检查所有依赖和硬件环境是否满足 v2.0 运行要求
"""
import sys

def check_python():
    v = sys.version_info
    ok = v.major == 3 and v.minor >= 8
    print(f"[{'✓' if ok else '✗'}] Python {v.major}.{v.minor}.{v.micro} {'(需要 3.8+)' if not ok else ''}")
    return ok

def check_opencv():
    try:
        import cv2
        ver = cv2.__version__
        # 检查 GStreamer 支持
        build_info = cv2.getBuildInformation()
        gstreamer = "YES" in build_info.split("GStreamer")[1].split("\n")[0] if "GStreamer" in build_info else False
        # 检查 CUDA 支持
        cuda = hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0
        cuda_count = cv2.cuda.getCudaEnabledDeviceCount() if hasattr(cv2, 'cuda') else 0
        
        print(f"[✓] OpenCV {ver}")
        print(f"  {'[✓]' if gstreamer else '[✗]'} GStreamer 支持 {'(MIPI CSI 需要)' if not gstreamer else ''}")
        print(f"  {'[✓]' if cuda else '[!]'} CUDA 支持 ({cuda_count} 设备) {'(可选，用于加速)' if not cuda else ''}")
        
        # SIFT 检查
        try:
            cv2.SIFT_create()
            print(f"  [✓] SIFT (需要 opencv-contrib)")
        except:
            print(f"  [✗] SIFT 不可用 (需要 opencv-contrib-python)")
        
        return True
    except ImportError:
        print("[✗] OpenCV 未安装")
        return False

def check_numpy():
    try:
        import numpy
        print(f"[✓] NumPy {numpy.__version__}")
        return True
    except:
        print("[✗] NumPy 未安装")
        return False

def check_mipi():
    """检查 MIPI CSI 摄像头"""
    import subprocess
    try:
        result = subprocess.run(["gst-inspect-1.0", "nvarguscamerasrc"], 
                               capture_output=True, text=True, timeout=5)
        ok = result.returncode == 0
        print(f"[{'✓' if ok else '✗'}] nvarguscamerasrc {'可用' if ok else '不可用'}")
        return ok
    except:
        print("[!] 无法检测 nvarguscamerasrc (非 Jetson 平台?)")
        return False

def check_jetson():
    """检查 Jetson 平台信息"""
    try:
        with open("/etc/nv_tegra_release") as f:
            info = f.readline().strip()
        print(f"[✓] Jetson 平台: {info}")
        return True
    except:
        print("[!] 非 Jetson 平台（MIPI CSI 不可用，可使用 RTSP 模式）")
        return False

def main():
    print("=" * 50)
    print("A8mini Tracker v2.0 环境检测")
    print("=" * 50)
    
    results = []
    results.append(("Python", check_python()))
    results.append(("OpenCV", check_opencv()))
    results.append(("NumPy", check_numpy()))
    
    # PyYAML
    try:
        import yaml
        print(f"[✓] PyYAML {yaml.__version__}")
        results.append(("PyYAML", True))
    except:
        print("[!] PyYAML 未安装 (可选，config.yaml 有 fallback 解析)")
        results.append(("PyYAML", True))
    
    print()
    results.append(("Jetson", check_jetson()))
    results.append(("MIPI CSI", check_mipi()))
    
    print()
    print("=" * 50)
    ok = all(r[1] for r in results)
    if ok:
        print("环境检测通过! 可以运行 v2.0")
    else:
        failed = [r[0] for r in results if not r[1]]
        print(f"以下项目需要修复: {', '.join(failed)}")

if __name__ == "__main__":
    main()
