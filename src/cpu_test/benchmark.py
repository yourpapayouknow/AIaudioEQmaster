from __future__ import annotations

import time
from pathlib import Path

from ..cpu.mastering import Config as CpuConfig, master_audio as cpu_master_audio
from .mastering import Config as CpuTestConfig, master_audio as cpu_test_master_audio


def run_benchmark(input_file: str, out_dir: str, genre: str = "Pop", eq_style: str = "Fusion", loudness: str = "loud"):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cpu_out = str(out_path / "bench_cpu.wav")
    test_out = str(out_path / "bench_cpu_test.wav")

    cfg_cpu = CpuConfig()
    cfg_cpu.genre = genre
    cfg_cpu.loudness_option = loudness

    t0 = time.perf_counter()
    cpu_master_audio(input_file, cpu_out, cfg_cpu, eq_style, False)
    cpu_elapsed = time.perf_counter() - t0

    cfg_test = CpuTestConfig()
    cfg_test.genre = genre
    cfg_test.loudness_option = loudness

    t1 = time.perf_counter()
    cpu_test_master_audio(input_file, test_out, cfg_test, eq_style, False)
    test_elapsed = time.perf_counter() - t1

    speedup = cpu_elapsed / max(test_elapsed, 1e-8)
    return {
        "cpu_seconds": round(cpu_elapsed, 3),
        "cpu_test_seconds": round(test_elapsed, 3),
        "speedup_x": round(speedup, 3),
        "cpu_out": cpu_out,
        "cpu_test_out": test_out,
    }

