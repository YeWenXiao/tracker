"""
性能基准测试
测试识别引擎在不同模式下的性能，输出对比表格

用法：
  python benchmark.py                    # 默认测试
  python benchmark.py --image test.jpg   # 指定测试图片
  python benchmark.py --rounds 50        # 指定测试轮数
  python benchmark.py --no-cuda          # 强制禁用 CUDA 对比
"""

import cv2
import numpy as np
import argparse
import time
import os
import sys


def create_test_image(width=1280, height=720):
    """生成一张合成测试图（含随机纹理和色块）"""
    img = np.random.randint(60, 200, (height, width, 3), dtype=np.uint8)
    # 添加一些色块和纹理
    cv2.rectangle(img, (200, 150), (400, 350), (0, 120, 255), -1)
    cv2.rectangle(img, (600, 300), (900, 550), (255, 80, 0), -1)
    cv2.circle(img, (width // 2, height // 2), 100, (0, 255, 120), -1)
    # 高斯模糊让纹理更自然
    img = cv2.GaussianBlur(img, (5, 5), 1.0)
    return img


def run_benchmark(rec, image, rounds, label=""):
    """运行 benchmark，返回各模式的耗时统计"""
    results = {}

    for mode_name, fast in [("fast", True), ("full", False)]:
        timings_list = []
        for i in range(rounds):
            _, timing = rec.recognize(image, fast=fast)
            timings_list.append(timing)

        # 汇总统计
        all_keys = set()
        for t in timings_list:
            all_keys.update(t.keys())

        stats = {}
        for key in sorted(all_keys):
            vals = [t[key] * 1000 for t in timings_list if key in t]  # ms
            if vals:
                stats[key] = {
                    "mean": np.mean(vals),
                    "min": np.min(vals),
                    "max": np.max(vals),
                    "std": np.std(vals),
                }
        results[mode_name] = stats

    return results


def print_table(results, label=""):
    """格式化输出对比表格"""
    if label:
        print(f"\n{'=' * 70}")
        print(f"  {label}")
        print(f"{'=' * 70}")

    for mode_name in ["fast", "full"]:
        if mode_name not in results:
            continue
        stats = results[mode_name]
        mode_label = "快速模式 (FAST)" if mode_name == "fast" else "全量模式 (FULL)"
        print(f"\n  [{mode_label}]")
        print(f"  {'步骤':<16s} {'平均(ms)':>10s} {'最小':>8s} {'最大':>8s} {'标准差':>8s}")
        print(f"  {'-' * 54}")

        # 按固定顺序输出
        key_order = ["template", "orb", "sift", "color_bp", "edge", "nms_verify", "total"]
        for key in key_order:
            if key in stats:
                s = stats[key]
                marker = ">>" if key == "total" else "  "
                print(f"  {marker}{key:<14s} {s['mean']:>10.1f} {s['min']:>8.1f} "
                      f"{s['max']:>8.1f} {s['std']:>8.1f}")


def main():
    parser = argparse.ArgumentParser(description="识别引擎性能基准测试")
    parser.add_argument("--image", type=str, help="测试图片路径 (不指定则生成合成图)")
    parser.add_argument("--rounds", type=int, default=20, help="每种模式的测试轮数 (默认20)")
    parser.add_argument("--no-cuda", action="store_true", help="强制禁用 CUDA 进行对比测试")
    parser.add_argument("--targets-dir", default="targets", help="目标模板目录")
    args = parser.parse_args()

    # 加载测试图片
    if args.image:
        image = cv2.imread(args.image)
        if image is None:
            print(f"无法读取图片: {args.image}")
            sys.exit(1)
        img_desc = os.path.basename(args.image)
    else:
        image = create_test_image()
        img_desc = "合成测试图 (1280x720)"

    print(f"性能基准测试")
    print(f"  测试图片: {img_desc} ({image.shape[1]}x{image.shape[0]})")
    print(f"  测试轮数: {args.rounds}")

    # 检查 targets 目录
    if not os.path.isdir(args.targets_dir):
        print(f"\n[WARNING] 目标目录 '{args.targets_dir}' 不存在")
        print(f"请先准备目标模板，或指定 --targets-dir")
        sys.exit(1)

    from recognize import TargetRecognizer

    # ---- 正常模式 benchmark ----
    print(f"\n--- 加载识别引擎 ---")
    t0 = time.time()
    rec = TargetRecognizer(targets_dir=args.targets_dir)
    load_time = (time.time() - t0) * 1000
    cuda_status = "CUDA" if rec.use_cuda else "CPU"
    tmpl_status = "CUDA" if rec.cuda_template_matcher is not None else "CPU"
    print(f"  模板加载耗时: {load_time:.0f} ms")
    print(f"  运行模式: {cuda_status} (模板匹配: {tmpl_status})")
    print(f"  目标数量: {len(rec.targets)}")

    # warmup
    print(f"\n--- 预热 (3轮) ---")
    for _ in range(3):
        rec.recognize(image, fast=True)
    print("  完成")

    # benchmark
    print(f"\n--- 运行 benchmark ({args.rounds} 轮) ---")
    results_normal = run_benchmark(rec, image, args.rounds, label=cuda_status)
    print_table(results_normal, label=f"识别性能 [{cuda_status}] (模板匹配: {tmpl_status})")

    # ---- 如果有CUDA，可选对比CPU模式 ----
    if rec.use_cuda and not args.no_cuda:
        print(f"\n--- CPU 对比测试 ---")
        print(f"  (使用 --no-cuda 跳过此测试)")
        # 临时禁用 CUDA
        rec.use_cuda = False
        saved_matcher = rec.cuda_template_matcher
        rec.cuda_template_matcher = None
        results_cpu = run_benchmark(rec, image, args.rounds)
        print_table(results_cpu, label="识别性能 [CPU] (强制禁用CUDA)")

        # 恢复
        rec.use_cuda = True
        rec.cuda_template_matcher = saved_matcher

        # 加速比
        print(f"\n{'=' * 70}")
        print(f"  加速比 (CPU / CUDA)")
        print(f"{'=' * 70}")
        for mode in ["fast", "full"]:
            if mode in results_cpu and mode in results_normal:
                cpu_total = results_cpu[mode].get("total", {}).get("mean", 0)
                gpu_total = results_normal[mode].get("total", {}).get("mean", 0)
                if gpu_total > 0:
                    ratio = cpu_total / gpu_total
                    mode_label = "快速" if mode == "fast" else "全量"
                    print(f"  {mode_label}模式: {cpu_total:.1f}ms -> {gpu_total:.1f}ms "
                          f"(加速 {ratio:.2f}x)")

    print(f"\n测试完成。")


if __name__ == "__main__":
    main()
