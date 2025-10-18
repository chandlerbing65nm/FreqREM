#!/usr/bin/env python3
import argparse
import re
import sys
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Tuple

# Regex patterns for metrics
PATTERNS = {
    "Error": re.compile(r"Error % \[[^\]]+\]:\s*([\d.]+)%"),
    "NLL": re.compile(r"NLL \[[^\]]+\]:\s*([\d.]+)"),
    "ECE": re.compile(r"ECE \[[^\]]+\]:\s*([\d.]+)"),
    "Adaptation Time": re.compile(r"Adaptation Time .*?\[[^\]]+\]:\s*([\d.]+)s"),
    # MACs may appear as a float or like "5.548 x 10^14"
    "Adaptation MACs": re.compile(r"Adaptation MACs .*?\[[^\]]+\]:\s*(.+)$"),
    "Domain Shift Robustness": re.compile(r"Domain Shift Robustness .*?\[[^\]]+\]:\s*([\d.]+)"),
    "Catastrophic Forgetting Rate": re.compile(r"Catastrophic Forgetting Rate .*? after \[[^\]]+\]:\s*([\d.]+)")
}

SCIENTIFIC_MACS_RE = re.compile(r"^\s*([\d.]+)\s*[xX]\s*10\^([+\-]?\d+)\s*$")


def parse_macs_value(val_str: str) -> float:
    """Parse MACs value that may be like '5.548 x 10^14' or a plain float.
    Returns float, or raises ValueError if unparsable.
    """
    s = val_str.strip()
    m = SCIENTIFIC_MACS_RE.match(s)
    if m:
        base = float(m.group(1))
        exp = int(m.group(2))
        return base * (10 ** exp)
    # Try to extract a simple float if possible (e.g., '0.123' or with trailing tokens)
    # Find first float in the string
    m2 = re.search(r"([+-]?[0-9]*\.?[0-9]+)", s)
    if m2:
        return float(m2.group(1))
    raise ValueError(f"Unparsable MACs value: {val_str}")


def parse_log_file(path: Path) -> Dict[str, List[float]]:
    metrics: Dict[str, List[float]] = {
        "Error": [],
        "NLL": [],
        "ECE": [],
        "Adaptation Time": [],
        "Adaptation MACs": [],
        "Domain Shift Robustness": [],
        "Catastrophic Forgetting Rate": [],
    }
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"Error reading {path}: {e}", file=sys.stderr)
        return metrics

    for line in text.splitlines():
        for key, pattern in PATTERNS.items():
            m = pattern.search(line)
            if not m:
                continue
            try:
                if key == "Adaptation MACs":
                    val = parse_macs_value(m.group(1))
                else:
                    val = float(m.group(1))
                metrics[key].append(val)
            except Exception:
                # Skip unparsable lines for robustness
                pass
    return metrics


def summarize_metrics(metrics: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    # Convert certain metrics to percentages
    converted: Dict[str, List[float]] = {}
    for k, values in metrics.items():
        if not values:
            converted[k] = []
            continue
        if k in ("Domain Shift Robustness", "Catastrophic Forgetting Rate"):
            converted[k] = [v * 100.0 for v in values]
        else:
            converted[k] = list(values)

    summary: Dict[str, Dict[str, float]] = {}
    for k, values in converted.items():
        if values:
            m = float(mean(values))
            # stdev requires at least 2 data points
            s = float(stdev(values)) if len(values) > 1 else float("nan")
            summary[k] = {
                "count": float(len(values)),
                "mean": m,
                "std": s,
            }
        else:
            summary[k] = {
                "count": 0.0,
                "mean": float("nan"),
                "std": float("nan"),
            }
    return summary


def print_summary(file_path: Path, summary: Dict[str, Dict[str, float]], as_json: bool = False):
    if as_json:
        out = {
            "file": str(file_path),
            "summary": summary,
        }
        print(json.dumps(out, indent=2))
        return

    print(f"\nFile: {file_path}")
    
    def fmt_mean(v: float, decimals: int = 3) -> str:
        if v != v:  # NaN
            return "nan"
        s = f"{v:.{decimals}f}"
        # strip trailing zeros and dot to reflect "up to" decimals
        s = s.rstrip('0').rstrip('.') if '.' in s else s
        return s

    def fmt_std(v: float, decimals: int = 2) -> str:
        if v != v:  # NaN
            return "nan"
        s = f"{v:.{decimals}f}"
        # keep two decimals per requirement, don't strip aggressively here
        return s

    for metric in [
        "Error",
        "NLL",
        "ECE",
        "Adaptation Time",
        "Adaptation MACs",
        "Domain Shift Robustness",
        "Catastrophic Forgetting Rate",
    ]:
        s = summary.get(metric, {"count": 0.0, "mean": float("nan"), "std": float("nan")})
        count_str = int(s["count"]) if s.get("count") == s.get("count") else 0

        # Adaptation MACs: mean only, keep generic compact formatting
        if metric == "Adaptation MACs":
            mean_str = f"{s['mean']:.6g}" if s["mean"] == s["mean"] else "nan"
            print(f"  {metric:>28}: count={count_str:2d}, mean={mean_str}")
            continue

        # Other metrics: show MEAN (STD) with rounding rules
        mean_str = fmt_mean(s["mean"], 3)
        std_str = fmt_std(s["std"], 2)

        # Add % sign for specific metrics now represented as percentages
        if metric in ("Domain Shift Robustness", "Catastrophic Forgetting Rate"):
            mean_str = mean_str + "%"
            std_str = std_str + "%"

        print(f"  {metric:>28}: count={count_str:2d}, mean={mean_str} ({std_str})")


def write_csv(rows: List[Tuple[str, str, int, float]], csv_path: Path):
    try:
        with csv_path.open("w", encoding="utf-8") as f:
            f.write("file,metric,count,mean\n")
            for file, metric, count, mean_val in rows:
                f.write(f"{file},{metric},{count},{mean_val}\n")
    except Exception as e:
        print(f"Failed to write CSV {csv_path}: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Parse log files and compute per-metric averages.")
    parser.add_argument("logs", nargs="*", type=str, help="Paths to log files to parse")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of human-readable summary")
    parser.add_argument("--csv", type=str, default=None, help="Optional path to write CSV summary")
    args = parser.parse_args()

    # If no logs given, try default three mentioned paths (if they exist)
    default_logs = [
        "/users/doloriel/work/Repo/FreqREM/logs/output_13685926.txt",
        "/users/doloriel/work/Repo/FreqREM/logs/output_13685937.txt",
        "/users/doloriel/work/Repo/FreqREM/logs/output_13685946.txt",
    ]
    log_paths = [Path(p) for p in (args.logs if args.logs else default_logs)]

    csv_rows: List[Tuple[str, str, int, float]] = []

    for p in log_paths:
        if not p.exists():
            print(f"Warning: {p} does not exist, skipping.", file=sys.stderr)
            continue
        metrics = parse_log_file(p)
        summary = summarize_metrics(metrics)
        if args.csv:
            for metric, mvals in summary.items():
                csv_rows.append((str(p), metric, int(mvals["count"]), mvals["mean"]))
        print_summary(p, summary, as_json=args.json)

    if args.csv and csv_rows:
        write_csv(csv_rows, Path(args.csv))


if __name__ == "__main__":
    main()
