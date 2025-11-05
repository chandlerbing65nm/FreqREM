#!/usr/bin/env python3
import argparse
import re
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Tuple, DefaultDict
from collections import defaultdict

PATTERNS = {
    "Error": re.compile(r"Error % \[([^\]]+)\]:\s*([\d.]+)%"),
    "NLL": re.compile(r"NLL \[([^\]]+)\]:\s*([\d.]+)"),
    "ECE": re.compile(r"ECE \[([^\]]+)\]:\s*([\d.]+)"),
    "Adaptation Time": re.compile(r"Adaptation Time .*?\[([^\]]+)\]:\s*([\d.]+)s"),
    "Adaptation MACs": re.compile(r"Adaptation MACs .*?\[([^\]]+)\]:\s*(.+)$"),
    "Domain Shift Robustness": re.compile(r"Domain Shift Robustness .*?\[([^\]]+)\]:\s*([\d.]+)"),
    "Catastrophic Forgetting Rate": re.compile(r"Catastrophic Forgetting Rate .*? after \[([^\]]+)\]:\s*([\d.]+)")
}

SCIENTIFIC_MACS_RE = re.compile(r"^\s*([\d.]+)\s*[xX]\s*10\^([+\-]?\d+)\s*$")


def parse_macs_value(val_str: str) -> float:
    s = val_str.strip()
    m = SCIENTIFIC_MACS_RE.match(s)
    if m:
        base = float(m.group(1))
        exp = int(m.group(2))
        return base * (10 ** exp)
    m2 = re.search(r"([+-]?[0-9]*\\.?[0-9]+)", s)
    if m2:
        return float(m2.group(1))
    raise ValueError(f"Unparsable MACs value: {val_str}")


def parse_log_file_detailed(path: Path) -> Dict[str, List[Tuple[str, float]]]:
    out: Dict[str, List[Tuple[str, float]]] = {
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
        return out
    for line in text.splitlines():
        for key, pattern in PATTERNS.items():
            m = pattern.search(line)
            if not m:
                continue
            try:
                corr = m.group(1)
                if key == "Adaptation MACs":
                    val = parse_macs_value(m.group(2))
                else:
                    val = float(m.group(2))
                out[key].append((corr, val))
            except Exception:
                pass
    return out


def aggregate_across_logs(paths: List[Path]) -> Tuple[Dict[str, Dict[str, List[float]]], Dict[str, List[str]], Dict[str, Dict[str, List[float]]]]:
    metrics_order = [
        "Error",
        "NLL",
        "ECE",
        "Adaptation Time",
        "Adaptation MACs",
        "Domain Shift Robustness",
        "Catastrophic Forgetting Rate",
    ]
    agg: Dict[str, DefaultDict[str, List[float]]] = {
        m: defaultdict(list) for m in metrics_order
    }
    order_by_metric: Dict[str, List[str]] = {m: [] for m in metrics_order}
    # Per-log values across corruptions for each metric: per_log_vals[metric][log_path] = [vals across corruptions]
    per_log_vals: Dict[str, Dict[str, List[float]]] = {m: {} for m in metrics_order}
    for p in paths:
        if not p.exists():
            print(f"Warning: {p} does not exist, skipping.", file=sys.stderr)
            continue
        parsed = parse_log_file_detailed(p)
        for metric, pairs in parsed.items():
            for corr, val in pairs:
                if corr not in order_by_metric[metric]:
                    order_by_metric[metric].append(corr)
                agg[metric][corr].append(val)
                per_log_vals[metric].setdefault(str(p), []).append(val)
    return {m: dict(cmap) for m, cmap in agg.items()}, order_by_metric, per_log_vals


def convert_units(values_by_corr: Dict[str, List[float]], metric: str) -> Dict[str, List[float]]:
    if metric in ("Domain Shift Robustness", "Catastrophic Forgetting Rate"):
        return {k: [v * 100.0 for v in vs] for k, vs in values_by_corr.items()}
    return values_by_corr


def compute_stats(vals: List[float]) -> Tuple[int, float, float]:
    if not vals:
        return 0, float("nan"), float("nan")
    c = len(vals)
    m = float(mean(vals))
    s = float(stdev(vals)) if c > 1 else float("nan")
    return c, m, s


def fmt_mean(v: float, decimals: int = 3) -> str:
    if v != v:
        return "nan"
    s = f"{v:.{decimals}f}"
    s = s.rstrip('0').rstrip('.') if '.' in s else s
    return s


def fmt_std(v: float, decimals: int = 2) -> str:
    if v != v:
        return "nan"
    return f"{v:.{decimals}f}"


def print_aggregate(agg: Dict[str, Dict[str, List[float]]], order_by_metric: Dict[str, List[str]], per_log_vals: Dict[str, Dict[str, List[float]]]):
    order = [
        "Error",
        "NLL",
        "ECE",
        "Adaptation Time",
        "Adaptation MACs",
        "Domain Shift Robustness",
        "Catastrophic Forgetting Rate",
    ]
    for metric in order:
        values_by_corr = convert_units(agg.get(metric, {}), metric)
        print(f"\nMetric: {metric}")
        means_for_overall: List[float] = []
        order_list = order_by_metric.get(metric, [])
        # Ensure we include any keys that might not be in the recorded order (unlikely)
        corrs_iter = order_list + [c for c in values_by_corr.keys() if c not in order_list]
        for corr in corrs_iter:
            c, m, s = compute_stats(values_by_corr[corr])
            if metric in ("Domain Shift Robustness", "Catastrophic Forgetting Rate"):
                m_str = fmt_mean(m) + "%"
                s_str = fmt_std(s) + "%"
            elif metric == "Adaptation MACs":
                m_str = f"{m:.6g}" if m == m else "nan"
                s_str = f"{s:.6g}" if s == s else "nan"
            else:
                m_str = fmt_mean(m)
                s_str = fmt_std(s)
            print(f"  {corr:>28}: count={c:2d}, mean={m_str} ({s_str})")
            if m == m:
                means_for_overall.append(m)
        if means_for_overall:
            overall_mean = float(mean(means_for_overall))
            overall_std = float(stdev(means_for_overall)) if len(means_for_overall) > 1 else float("nan")
        else:
            overall_mean = float("nan")
            overall_std = float("nan")
        if metric in ("Domain Shift Robustness", "Catastrophic Forgetting Rate"):
            om = fmt_mean(overall_mean) + "%"
            os = fmt_std(overall_std) + "%"
        elif metric == "Adaptation MACs":
            om = f"{overall_mean:.6g}" if overall_mean == overall_mean else "nan"
            os = f"{overall_std:.6g}" if overall_std == overall_std else "nan"
        else:
            om = fmt_mean(overall_mean)
            os = fmt_std(overall_std)
        print(f"  {'Overall (across corruptions)':>28}: mean={om} ({os})")

        # Overall across logs: compute per-log mean/std across corruptions, then average these across logs
        log_series = per_log_vals.get(metric, {})
        per_log_means: List[float] = []
        per_log_stds: List[float] = []
        for lp, vals in log_series.items():
            conv_vals = convert_units({"_": vals}, metric)["_"]
            if not conv_vals:
                continue
            per_log_means.append(float(mean(conv_vals)))
            s_val = float(stdev(conv_vals)) if len(conv_vals) > 1 else float("nan")
            per_log_stds.append(s_val)
        if per_log_means:
            overall_mean_logs = float(mean(per_log_means))
            overall_std_logs = float(stdev(per_log_means)) if len(per_log_means) > 1 else float("nan")
        else:
            overall_mean_logs = float("nan")
            overall_std_logs = float("nan")

        if metric in ("Domain Shift Robustness", "Catastrophic Forgetting Rate"):
            om_logs = fmt_mean(overall_mean_logs) + "%"
            os_logs = fmt_std(overall_std_logs) + "%"
        elif metric == "Adaptation MACs":
            om_logs = f"{overall_mean_logs:.6g}" if overall_mean_logs == overall_mean_logs else "nan"
            os_logs = f"{overall_std_logs:.6g}" if overall_std_logs == overall_std_logs else "nan"
        else:
            om_logs = fmt_mean(overall_mean_logs)
            os_logs = fmt_std(overall_std_logs)
        print(f"  {'Overall (across logs)':>28}: mean={om_logs} ({os_logs})")


def main():
    parser = argparse.ArgumentParser(description="Aggregate corruption-wise metrics across multiple logs.")
    parser.add_argument("logs", nargs="*", type=str, help="Paths to log files to parse")
    args = parser.parse_args()

    default_logs = [
        "/users/doloriel/work/Repo/SPARC/logs/output_13685926.txt",
        "/users/doloriel/work/Repo/SPARC/logs/output_13685937.txt",
        "/users/doloriel/work/Repo/SPARC/logs/output_13685946.txt",
    ]
    log_paths = [Path(p) for p in (args.logs if args.logs else default_logs)]

    agg, order_by_metric, per_log_vals = aggregate_across_logs(log_paths)
    print_aggregate(agg, order_by_metric, per_log_vals)


if __name__ == "__main__":
    main()
