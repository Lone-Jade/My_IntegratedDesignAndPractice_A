#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TCRT5000 串口预测结果显示器
==========================
读取单片机串口输出，解析板载 AI 推理结果行（各数字预测概率），
以终端条形图显示当前预测，并保留最近若干次的历史记录。

单片机端（Core/Src/main.c，ST Edge AI int8 网络）每扫完一个数字输出一行：
    [AI] rows=56 best=9:94% | 0:0% 1:0% 2:0% 3:0% 4:0% 5:1% 6:0% 7:0% 8:5% 9:94%
其中 rows = 扫描行数（裁剪前），best = 置信最高的数字，后面是 10 个数字的概率(%)。

串口里同时还有原始二进制帧（AA 55 ...，PC 端采集工具用），本程序自动忽略它们，
只按 "[AI]" 标记定位文本结果行，因此二进制帧与文本结果可以同时输出。

用法:
    python3 tools/show_prediction.py /dev/ttyUSB0                 # Linux，默认 115200
    python3 tools/show_prediction.py COM3                         # Windows
    python3 tools/show_prediction.py /dev/ttyUSB0 --baud 115200 --history 20
    python3 tools/show_prediction.py /dev/ttyUSB0 --no-clear      # 不整屏刷新，顺序打印

依赖: pyserial（pip install pyserial）
"""
import argparse
import re
import sys
import time

try:
    import serial
except ImportError:
    sys.exit("缺少 pyserial：pip install pyserial")

RESULT_RE = re.compile(r"\[AI\] rows=(\d+) best=(\d+):(\d+)%")
PROB_RE = re.compile(r"(\d+):(\d+)%")
BAR_WIDTH = 50          # 条形图最大宽度（字符）
KEEP_TAIL = 512         # 半行缓冲保留字节数


def parse_result(line):
    """从一行 '[AI] ...' 中解析 {rows, best, probs[10]}；不是结果行返回 None。"""
    m = RESULT_RE.search(line)
    if not m:
        return None
    rows = int(m.group(1))
    best = int(m.group(2))
    probs = [0] * 10
    for d, p in PROB_RE.findall(line):
        d = int(d)
        if 0 <= d <= 9:
            probs[d] = int(p)
    return {"rows": rows, "best": best, "probs": probs}


def extract_results(buf):
    """从混合字节流中提取全部完整的结果行。返回 (results, 剩余缓冲)。"""
    results = []
    while True:
        i = buf.find(b"[AI]")
        if i < 0:
            return results, buf[-KEEP_TAIL:]
        end = buf.find(b"\n", i)
        if end < 0:                     # 行未收完，等下一次数据
            return results, buf[i:]
        line = buf[i:end].decode(errors="replace").strip()
        buf = buf[end + 1:]
        r = parse_result(line)
        if r:
            results.append(r)


def render(result, history, no_clear):
    """生成当前结果的终端显示文本。"""
    if not no_clear and sys.stdout.isatty():
        sys.stdout.write("\x1b[2J\x1b[H")          # ANSI 整屏刷新
    out = []
    out.append("=" * 58)
    out.append("  TCRT5000 板载识别结果")
    out.append(f"  扫描行数: {result['rows']}")
    out.append("")
    best = result["best"]
    for d in range(10):
        p = result["probs"][d]
        bar = "#" * int(round(p / 100.0 * BAR_WIDTH))
        mark = ">>" if d == best else "  "
        out.append(f"  {mark} {d}: {p:3d}% |{bar}")
    out.append("")
    out.append(f"  ==> 预测数字: {best}   (置信 {result['probs'][best]}%)")
    if history:
        hist = "  ".join(f"{r['best']}({r['probs'][r['best']]}%)" for r in history)
        out.append(f"  最近结果: {hist}")
    out.append("=" * 58)
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(
        description="TCRT5000 串口预测结果显示（解析 [AI] 结果行）")
    ap.add_argument("port", help="串口，如 /dev/ttyUSB0 或 COM3")
    ap.add_argument("-b", "--baud", type=int, default=115200)
    ap.add_argument("--history", type=int, default=8,
                    help="历史记录条数（默认 8）")
    ap.add_argument("--no-clear", action="store_true",
                    help="不整屏刷新，每次结果顺序打印")
    args = ap.parse_args()

    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.1)
    except serial.SerialException as e:
        sys.exit(f"无法打开串口 {args.port}: {e}")

    print(f"监听 {args.port} @ {args.baud}，等待板载推理结果..."
          f"（扫描数字后拿开纸片，每扫一个输出一次，Ctrl+C 退出）\n")
    buf = b""
    history = []
    try:
        while True:
            data = ser.read(4096)
            if not data:
                continue
            buf += data
            results, buf = extract_results(buf)
            for r in results:
                history.append(r)
                del history[:-args.history]
                sys.stdout.write(render(r, history, args.no_clear) + "\n")
                sys.stdout.flush()
    except KeyboardInterrupt:
        print("\n--- 已停止 ---")
    finally:
        ser.close()


if __name__ == "__main__":
    main()
