#!/usr/bin/env python3
"""
HADM Performance Monitor
Tracks import times, GPU usage, and system performance
"""

import time
import sys
import os
import psutil
import subprocess
import threading
from pathlib import Path
from typing import Dict, List, Optional
import json


class PerformanceMonitor:
    """Monitor system and application performance"""

    def __init__(self):
        self.start_time = time.time()
        self.import_times = {}
        self.gpu_stats = []
        self.system_stats = []
        self.monitoring = False
        self.monitor_thread = None

    def start_monitoring(self, interval: float = 1.0):
        """Start background monitoring"""
        if self.monitoring:
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop, args=(interval,), daemon=True
        )
        self.monitor_thread.start()
        print(f"[MONITOR] Started performance monitoring (interval: {interval}s)")

    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        print("[MONITOR] Stopped performance monitoring")

    def _monitor_loop(self, interval: float):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                # Collect system stats
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()

                # Collect GPU stats
                gpu_info = self._get_gpu_stats()

                timestamp = time.time() - self.start_time

                self.system_stats.append(
                    {
                        "timestamp": timestamp,
                        "cpu_percent": cpu_percent,
                        "memory_percent": memory.percent,
                        "memory_used_gb": memory.used / (1024**3),
                        "memory_available_gb": memory.available / (1024**3),
                    }
                )

                if gpu_info:
                    gpu_info["timestamp"] = timestamp
                    self.gpu_stats.append(gpu_info)

                time.sleep(interval)

            except Exception as e:
                print(f"[MONITOR] Error in monitoring loop: {e}")
                time.sleep(interval)

    def _get_gpu_stats(self) -> Optional[Dict]:
        """Get GPU statistics using nvidia-smi"""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                if lines and lines[0]:
                    values = lines[0].split(", ")
                    if len(values) >= 4:
                        return {
                            "memory_used_mb": int(values[0]),
                            "memory_total_mb": int(values[1]),
                            "gpu_utilization": int(values[2]),
                            "temperature": int(values[3]),
                        }
        except Exception:
            pass
        return None

    def log_import_time(self, module_name: str, duration: float):
        """Log import timing"""
        self.import_times[module_name] = duration
        print(f"[IMPORT] {module_name}: {duration:.2f}s")

    def time_import(self, module_name: str):
        """Context manager for timing imports"""
        return ImportTimer(self, module_name)

    def get_summary(self) -> Dict:
        """Get performance summary"""
        total_runtime = time.time() - self.start_time

        summary = {
            "total_runtime": total_runtime,
            "import_times": self.import_times,
            "total_import_time": sum(self.import_times.values()),
            "slowest_imports": sorted(
                self.import_times.items(), key=lambda x: x[1], reverse=True
            )[:5],
        }

        if self.system_stats:
            latest_system = self.system_stats[-1]
            summary["current_system"] = {
                "cpu_percent": latest_system["cpu_percent"],
                "memory_percent": latest_system["memory_percent"],
                "memory_used_gb": latest_system["memory_used_gb"],
            }

            # Calculate averages
            summary["average_system"] = {
                "cpu_percent": sum(s["cpu_percent"] for s in self.system_stats)
                / len(self.system_stats),
                "memory_percent": sum(s["memory_percent"] for s in self.system_stats)
                / len(self.system_stats),
            }

        if self.gpu_stats:
            latest_gpu = self.gpu_stats[-1]
            summary["current_gpu"] = {
                "memory_used_mb": latest_gpu["memory_used_mb"],
                "memory_total_mb": latest_gpu["memory_total_mb"],
                "memory_percent": (
                    latest_gpu["memory_used_mb"] / latest_gpu["memory_total_mb"]
                )
                * 100,
                "gpu_utilization": latest_gpu["gpu_utilization"],
                "temperature": latest_gpu["temperature"],
            }

            # Calculate averages
            summary["average_gpu"] = {
                "memory_percent": sum(
                    (s["memory_used_mb"] / s["memory_total_mb"]) * 100
                    for s in self.gpu_stats
                )
                / len(self.gpu_stats),
                "gpu_utilization": sum(s["gpu_utilization"] for s in self.gpu_stats)
                / len(self.gpu_stats),
            }

        return summary

    def print_summary(self):
        """Print performance summary"""
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print("PERFORMANCE SUMMARY")
        print("=" * 60)

        print(f"Total Runtime: {summary['total_runtime']:.2f}s")
        print(f"Total Import Time: {summary['total_import_time']:.2f}s")
        print(
            f"Import Overhead: {(summary['total_import_time']/summary['total_runtime']*100):.1f}%"
        )

        print("\nSlowest Imports:")
        for module, duration in summary["slowest_imports"]:
            print(f"  {module}: {duration:.2f}s")

        if "current_system" in summary:
            sys_stats = summary["current_system"]
            print(f"\nCurrent System Usage:")
            print(f"  CPU: {sys_stats['cpu_percent']:.1f}%")
            print(
                f"  Memory: {sys_stats['memory_percent']:.1f}% ({sys_stats['memory_used_gb']:.1f}GB)"
            )

        if "current_gpu" in summary:
            gpu_stats = summary["current_gpu"]
            print(f"\nCurrent GPU Usage:")
            print(
                f"  Memory: {gpu_stats['memory_percent']:.1f}% ({gpu_stats['memory_used_mb']}MB/{gpu_stats['memory_total_mb']}MB)"
            )
            print(f"  Utilization: {gpu_stats['gpu_utilization']}%")
            print(f"  Temperature: {gpu_stats['temperature']}°C")

        print("=" * 60)

    def save_report(self, filename: str = None):
        """Save detailed performance report"""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"performance_report_{timestamp}.json"

        report = {
            "summary": self.get_summary(),
            "detailed_system_stats": self.system_stats[-100:],  # Last 100 entries
            "detailed_gpu_stats": self.gpu_stats[-100:],  # Last 100 entries
            "all_import_times": self.import_times,
        }

        with open(filename, "w") as f:
            json.dump(report, f, indent=2)

        print(f"[MONITOR] Performance report saved to {filename}")
        return filename


class ImportTimer:
    """Context manager for timing imports"""

    def __init__(self, monitor: PerformanceMonitor, module_name: str):
        self.monitor = monitor
        self.module_name = module_name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.monitor.log_import_time(self.module_name, duration)


def analyze_import_performance():
    """Analyze import performance of key modules"""
    print("=" * 60)
    print("IMPORT PERFORMANCE ANALYSIS")
    print("=" * 60)

    monitor = PerformanceMonitor()
    monitor.start_monitoring(interval=0.5)

    # Test imports
    modules_to_test = [
        ("os", "import os"),
        ("sys", "import sys"),
        ("time", "import time"),
        ("pathlib", "from pathlib import Path"),
        ("torch", "import torch"),
        ("numpy", "import numpy as np"),
        ("PIL", "from PIL import Image"),
        ("fastapi", "from fastapi import FastAPI"),
        ("uvicorn", "import uvicorn"),
        ("detectron2.config", "from detectron2.config import LazyConfig"),
    ]

    for module_name, import_statement in modules_to_test:
        try:
            with monitor.time_import(module_name):
                exec(import_statement)
        except ImportError as e:
            print(f"[IMPORT] {module_name}: FAILED - {e}")
        except Exception as e:
            print(f"[IMPORT] {module_name}: ERROR - {e}")

    # Wait a bit for system stats
    time.sleep(2)

    monitor.stop_monitoring()
    monitor.print_summary()
    monitor.save_report()


def monitor_server_startup(script_path: str = "api.py", lazy: bool = False):
    """Monitor server startup performance"""
    print("=" * 60)
    print("SERVER STARTUP MONITORING")
    print("=" * 60)

    monitor = PerformanceMonitor()
    monitor.start_monitoring(interval=0.1)  # High frequency monitoring

    # Start the server process
    cmd = [sys.executable, script_path]
    if lazy:
        cmd.append("--lazy")

    print(f"Starting server: {' '.join(cmd)}")

    try:
        # Monitor for 60 seconds or until process ends
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        start_time = time.time()
        while time.time() - start_time < 60:
            if process.poll() is not None:
                break
            time.sleep(0.1)

        # Terminate if still running
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)

    except KeyboardInterrupt:
        print("\nMonitoring interrupted by user")
        if "process" in locals():
            process.terminate()

    monitor.stop_monitoring()
    monitor.print_summary()

    report_file = monitor.save_report("server_startup_report.json")
    print(f"\nDetailed report saved to: {report_file}")


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="HADM Performance Monitor")
    parser.add_argument(
        "--mode",
        choices=["imports", "startup"],
        default="imports",
        help="Monitoring mode",
    )
    parser.add_argument(
        "--script", default="api.py", help="Script to monitor for startup mode"
    )
    parser.add_argument(
        "--lazy", action="store_true", help="Use lazy mode for startup monitoring"
    )

    args = parser.parse_args()

    if args.mode == "imports":
        analyze_import_performance()
    elif args.mode == "startup":
        monitor_server_startup(args.script, args.lazy)


if __name__ == "__main__":
    main()
