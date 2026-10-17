"""Uniform reporting for the scripts: every check is printed as [ok]/[FAIL] and collected
into a JSON file out/<step>.json together with the software versions and the run time.
スクリプト共通の報告: 各検査を [ok]/[FAIL] で表示し、ソフトウェアの版と実行時間とともに out/<step>.json にまとめる。
"""
import json
import os
import platform
import sys
import time

try:  # numpy is optional here (v2_theta.py runs without it) / ここでは numpy は任意（v2_theta.py は numpy なしで動く）
    import numpy as np
except ImportError:
    np = None

DEFAULT_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "out")


def _plain(o):
    """JSON fallback for numpy/Sage numbers: integers as int, everything else as str.
    numpy・Sage の数のための JSON の代替: 整数は int、それ以外は文字列。"""
    try:
        if int(o) == o:
            return int(o)
    except (TypeError, ValueError):
        pass
    return str(o)


class Report:
    def __init__(self, step, out_dir=None):
        self.step = step
        self.out_dir = out_dir or DEFAULT_OUT
        self.t0 = time.time()
        self.checks = []
        self.values = {}
        self.failed = False

    def log(self, *args):
        print(f"[{time.time() - self.t0:8.1f} s]", *args, flush=True)

    def check(self, label, ok, stop=True):
        """Record a yes/no check; by default stop at the first failure.
        はい・いいえの検査を記録する。既定では最初の失敗で止める。"""
        ok = bool(ok)
        self.checks.append({"check": label, "ok": ok})
        self.log("[ok]  " if ok else "[FAIL]", label)
        if not ok:
            self.failed = True
            if stop:
                self.finish()
                sys.exit(1)

    def value(self, key, v):
        """Record a computed value (number, list, ...). / 計算した値を記録する。"""
        self.values[key] = v
        self.log(f"{key} = {v}")

    def finish(self):
        os.makedirs(self.out_dir, exist_ok=True)
        result = {
            "step": self.step,
            "passed": not self.failed,
            "checks": self.checks,
            "values": self.values,
            "seconds": round(time.time() - self.t0, 1),
            "finished": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            "python": platform.python_version(),
            "numpy": np.__version__ if np is not None else None,
            "platform": platform.platform(),
        }
        path = os.path.join(self.out_dir, self.step + ".json")
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(result, f, indent=1, default=_plain)
            f.write("\n")
        os.replace(tmp, path)
        self.log(("PASSED" if not self.failed else "FAILED") + f": {self.step} -> {path}")
