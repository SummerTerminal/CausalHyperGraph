"""
ablation_runner.py - ablation experiments for paper Sec 4.4
"""

import os, json, argparse, subprocess, sys, re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class AblationConfig:
    """config for one ablation experiment"""
    name: str
    description: str
    flags: List[str]
    is_baseline: bool = False


class AblationRunner:
    """runner for ablation studies"""

    def __init__(self, base_command: List[str]):
        self.base_cmd = base_command
        self.res = {}  # store results later

    def run_single(self, config: AblationConfig, extra: Dict[str, str] = None, timeout=7200):
        """
        run one experiment, returns (accuracy, output string)
        """
        cmd = self.base_cmd.copy()
        # add ablation flags
        for f in config.flags:
            if '=' in f:
                cmd.append(f)
            else:
                cmd.append(f)
        # extra key-value arguments
        if extra:
            for k, v in extra.items():
                cmd += [f'--{k}', str(v)]

        print(f"\n{'=' * 60}\nRunning: {config.name}\nCommand: {' '.join(cmd)}\n{'=' * 60}")

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=os.environ.copy())
            out = proc.stdout + proc.stderr
            acc = self._extract_acc(out)
            return acc, out
        except subprocess.TimeoutExpired:
            print("  ⏰ timed out")
            return 0.0, "timeout :("
        except Exception as e:
            print(f"  ❌ error: {e}")
            return 0.0, str(e)

    def _extract_acc(self, output: str) -> float:
        """parse accuracy from log, try several patterns"""
        # 中文输出也得处理
        pats = [
            r'Accuracy.*?(\d+\.?\d*)\%',
            r'准确率.*?(\d+\.?\d*)\%',
            r'Best.*?(\d+\.?\d*)\%',
            r'Val Acc.*?(\d+\.?\d*)\%',
            r'val_acc.*?(\d+\.?\d*)'
        ]
        for p in pats:
            m = re.search(p, output, re.IGNORECASE)
            if m:
                return float(m.group(1))
        return 0.0

    def run_all(self, configs: List[AblationConfig], extra: Dict[str, str] = None):
        """run a list of experiments, save & return results"""
        out_dict = {}
        for cfg in configs:
            acc, output = self.run_single(cfg, extra)
            out_dict[cfg.name] = {
                'accuracy': acc,
                'description': cfg.description,
                'flags': cfg.flags,
                'is_baseline': cfg.is_baseline,
                'output_snippet': output[:500]  # keep it short
            }
            print(f"\n  📊 {cfg.name}: {acc:.2f}%")
        self.res = out_dict
        return out_dict

    def make_report(self, results: Dict[str, Dict] = None) -> str:
        """generate a markdown-style ablation report"""
        if results is None:
            results = self.res
        lines = []
        lines.append("\n" + "=" * 60)
        lines.append("Ablation Study Results")
        lines.append("=" * 60)

        # find baseline config
        base_name = None
        for n, d in results.items():
            if d.get('is_baseline'):
                base_name = n
                break
        if base_name:
            base_acc = results[base_name]['accuracy']
            lines.append(f"\nBaseline ({base_name}): {base_acc:.2f}%")
            lines.append("\nComparison:")
            for n, d in results.items():
                if n == base_name:
                    continue
                diff = d['accuracy'] - base_acc
                sign = "+" if diff > 0 else ""
                lines.append(f"  {n}: {d['accuracy']:.2f}% ({sign}{diff:.2f}%)")

        lines.append("\n" + "=" * 60)
        lines.append("\n| Config | Accuracy | Δ Baseline |")
        lines.append("|:---|:---|:---|")
        for n, d in results.items():
            acc = d['accuracy']
            if n == base_name:
                lines.append(f"| **{n}** | **{acc:.1f}%** | - |")
            else:
                diff = acc - results[base_name]['accuracy']
                lines.append(f"| {n} | {acc:.1f}% | {diff:+.1f}% |")
        return "\n".join(lines)


def get_paper_ablations():
    """configs copied from paper Section 4.4"""
    return [
        AblationConfig(
            name="BERT baseline (no graph)",
            description="just BERT, no graph structure",
            flags=["--no_hypergraph"],
            is_baseline=True
        ),
        AblationConfig(
            name="+ Plain graph (PlotTree)",
            description="use plain graph instead of hypergraph",
            flags=["--use_graph", "--no_hyperedge"],
            is_baseline=False
        ),
        AblationConfig(
            name="+ Hypergraph (no types)",
            description="hypergraph without type awareness",
            flags=["--no_type_aware"],
            is_baseline=False
        ),
        AblationConfig(
            name="+ Type-aware HGNE",
            description="type-aware hypergraph convolution",
            flags=["--type_aware"],
            is_baseline=False
        ),
        AblationConfig(
            name="+ HAR (hyperedge-aware)",
            description="hyperedge-aware retrieval",
            flags=["--use_har"],
            is_baseline=False
        ),
        AblationConfig(
            name="+ Pretrain+finetune (full)",
            description="complete model",
            flags=["--use_pretrain", "--use_finetune", "--use_har", "--type_aware"],
            is_baseline=False
        )
    ]


def main():
    parser = argparse.ArgumentParser(description='ablation experiment runner')
    parser.add_argument('--script', default='experiments/har/run_qa.py')
    parser.add_argument('--questions', required=True)
    parser.add_argument('--hypergraph_dir', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output_dir', default='experiments/ablation_results')
    parser.add_argument('--dataset', default='storyvideoqa_g')
    parser.add_argument('--dry_run', action='store_true', help='just print commands')

    a = parser.parse_args()

    base_cmd = [
        'python', a.script,
        '--questions', a.questions,
        '--hypergraph_dir', a.hypergraph_dir,
        '--checkpoint', a.checkpoint,
        '--output', os.path.join(a.output_dir, '${name}_result.json'),  # will be replaced later
        '--model_type', 'e2e'
    ]

    extra = {'dataset': a.dataset}

    if a.dry_run:
        print("dry run - printing commands only")
        for cfg in get_paper_ablations():
            cmd = base_cmd.copy()
            # replace placeholder with sanitized name
            cmd = [c.replace('${name}', cfg.name.replace(' ', '_')) for c in cmd]
            cmd += cfg.flags
            print(f"\n{cfg.name}: {' '.join(cmd)}")
        return

    runner = AblationRunner(base_cmd)
    res = runner.run_all(get_paper_ablations(), extra)

    os.makedirs(a.output_dir, exist_ok=True)
    with open(os.path.join(a.output_dir, 'ablation_results.json'), 'w') as f:
        json.dump(res, f, indent=2)

    report = runner.make_report(res)
    print(report)
    with open(os.path.join(a.output_dir, 'ablation_report.txt'), 'w') as f:
        f.write(report)


if __name__ == '__main__':
    main()