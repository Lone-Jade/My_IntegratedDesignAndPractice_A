#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TCRT5000 字符扫描数据采集器
============================
- 解码固件二进制帧（每行 21 字节）：AA 55 | 行号u16(LE) | 8×u16(LE) | 校验和u8
- 自动分段：有纸片扫过（任一通道 < 4095-裕量）视为"正在扫描一个字符"；
  数据恢复 4095（连续若干行）视为该字符扫完，保存为一个 CSV
- 文件名：<启动时间>-X.csv （X 为字符占位，事后手动重命名成 0/1/...）
  同一会话第 2 个字符起：<启动时间>-X-2.csv、-X-3.csv ...

用法:
    python3 scan_view.py /dev/ttyUSB0                  # 采集 + 自动保存 CSV
    python3 scan_view.py /dev/ttyUSB0 --image          # 额外实时显示 ASCII 图像
    python3 scan_view.py /dev/ttyUSB0 --margin 100     # 空闲判定裕量（默认 100）
    python3 scan_view.py /dev/ttyUSB0 --idle-rows 5    # 连续空闲行数判定段结束（默认 5）
    python3 scan_view.py /dev/ttyUSB0 --min-rows 3     # 段的最少行数，防误触发（默认 3）
    python3 scan_view.py /dev/ttyUSB0 --out-dir data   # CSV 保存目录（默认 outputs/）

依赖: pip install pyserial
"""
import sys
import time
import argparse
import datetime
import os

FRAME_LEN = 21
IDLE_FULL = 4095


def parse_args():
    ap = argparse.ArgumentParser(description="TCRT5000 character scanner collector")
    ap.add_argument("port", help="serial port, e.g. /dev/ttyUSB0")
    ap.add_argument("-b", "--baud", type=int, default=115200)
    ap.add_argument("--image", action="store_true", help="live render current segment")
    ap.add_argument("--thresh", type=int, default=2000, help="image binarize threshold")
    ap.add_argument("--invert", action="store_true", help="invert image polarity")
    ap.add_argument("--margin", type=int, default=100,
                    help="idle margin: row is idle when all channels >= 4095-margin (default 100)")
    ap.add_argument("--idle-rows", type=int, default=5,
                    help="consecutive idle rows to end a segment (default 5)")
    ap.add_argument("--min-rows", type=int, default=3,
                    help="minimum rows for a valid segment (default 3)")
    ap.add_argument("--out-dir", default="outputs",
                    help="directory to save CSV files (default: outputs)")
    return ap.parse_args()


class FrameDecoder:
    """从字节流中同步并解析 21 字节帧。"""

    def __init__(self):
        self.buf = bytearray()

    def feed(self, data):
        self.buf += data
        frames = []
        i = 0
        while i < len(self.buf):
            if self.buf[i] != 0xAA:          # 同步帧头
                i += 1
                continue
            if i + 1 >= len(self.buf):
                break
            if self.buf[i + 1] != 0x55:
                i += 1
                continue
            if i + FRAME_LEN > len(self.buf):
                break
            frame = bytes(self.buf[i:i + FRAME_LEN])
            csum = sum(frame[2:FRAME_LEN - 1]) & 0xFF
            if csum == frame[FRAME_LEN - 1]:
                row = int.from_bytes(frame[2:4], "little")
                ch = [int.from_bytes(frame[4 + 2 * k:6 + 2 * k], "little") for k in range(8)]
                frames.append((row, ch))
                i += FRAME_LEN
            else:
                i += 1
        del self.buf[:i]
        return frames


class Segmenter:
    """按 '4095 空闲' 把行流切分成字符段。push() 返回事件字符串。"""

    def __init__(self, margin=100, idle_rows=5, min_rows=3):
        self.margin = margin
        self.idle_rows = idle_rows
        self.min_rows = min_rows
        self.state = "idle"      # idle | active
        self.idle_count = 0
        self.segment = []        # [(row, ch), ...]

    def row_is_active(self, ch):
        return min(ch) < IDLE_FULL - self.margin

    def push(self, row, ch):
        """处理一行，返回 ('idle'|'active'|'segment_end', segment_or_None)"""
        if self.state == "idle":
            if self.row_is_active(ch):
                self.state = "active"
                self.idle_count = 0
                self.segment = [(row, ch)]
                return "active", None
            return "idle", None

        # active
        if self.row_is_active(ch):
            self.idle_count = 0
            self.segment.append((row, ch))
            return "active", None
        else:
            self.idle_count += 1
            if self.idle_count >= self.idle_rows:
                seg = self.segment
                self.state = "idle"
                self.idle_count = 0
                self.segment = []
                if len(seg) < self.min_rows:
                    return "segment_end", None   # 太短，忽略
                return "segment_end", seg
            return "active", None   # 仍在段中（短暂空闲未确认）


def render_image(rows, thresh, invert):
    print(f"  --- 当前段 {len(rows)} 行 ---")
    for _, ch in rows:
        line = ""
        for v in ch:
            b = v < thresh
            if invert:
                b = not b
            line += "#" if b else "."
        print(line)


def save_segment(segment, start_str, seg_index, out_dir):
    """segment: [(row, ch)]，保存为 <start_str>-X[-n].csv"""
    suffix = f"-{seg_index}"
    fname = f"{start_str}-X{suffix}.csv"
    fpath = os.path.join(out_dir, fname)
    with open(fpath, "w", newline="") as f:
        f.write("row,ch0,ch1,ch2,ch3,ch4,ch5,ch6,ch7\n")
        for row, ch in segment:
            f.write(f"{row}," + ",".join(str(v) for v in ch) + "\n")
    rows0, rows1 = segment[0][0], segment[-1][0]
    print(f"  ✅ 字符 {seg_index} 保存 {len(segment)} 行 (R{rows0}~R{rows1}) → {fpath}")
    print(f"     （把文件名里的 X 重命名为实际字符，如 ...-0.csv）")


def main():
    args = parse_args()
    try:
        import serial
    except ImportError:
        sys.exit("缺少 pyserial：pip install pyserial")
    ser = serial.Serial(args.port, args.baud, timeout=0.1)
    os.makedirs(args.out_dir, exist_ok=True)

    start = datetime.datetime.now()
    start_str = start.strftime("%Y-%m-%d-%H-%M-%S")
    print(f"=== TCRT5000 字符扫描采集器 ===")
    print(f"启动时间: {start_str}  (CSV 文件名前缀)")
    print(f"空闲判定: 全部通道 >= {IDLE_FULL - args.margin}; 段结束需连续 {args.idle_rows} 行空闲")
    print("扫描一个字符 → 拿开纸片等数据恢复 4095 → 再扫下一个字符；Ctrl+C 结束\n")

    dec = FrameDecoder()
    seg = Segmenter(args.margin, args.idle_rows, args.min_rows)
    seg_index = 0
    cur_segment_rows = []      # 供 --image 渲染
    last_render = 0.0

    try:
        while True:
            data = ser.read(256)
            if not data:
                continue
            for row, ch in dec.feed(data):
                print(f"R{row:5d}: " + " ".join(f"{v:4d}" for v in ch))
                ev, result = seg.push(row, ch)
                if ev == "active":
                    cur_segment_rows.append((row, ch))
                    if args.image and time.time() - last_render > 0.25:
                        render_image(cur_segment_rows, args.thresh, args.invert)
                        last_render = time.time()
                elif ev == "segment_end":
                    if result is not None:
                        seg_index += 1
                        save_segment(result, start_str, seg_index, args.out_dir)
                    else:
                        print("  (该段行数过少，忽略)")
                    cur_segment_rows = []
                    print("  <<< 数据已恢复 4095，等待下一个字符...\n")
    except KeyboardInterrupt:
        if seg.state == "active" and len(seg.segment) >= args.min_rows:
            seg_index += 1
            save_segment(seg.segment, start_str, seg_index, args.out_dir)
        print("--- 采集结束 ---")
    finally:
        ser.close()


if __name__ == "__main__":
    main()
