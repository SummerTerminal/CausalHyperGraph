"""
ablation_runner.py - 消融实验自动化
用于论文 4.4 节的消融实验
"""

import os
import json
import argparse
import subprocess
import sys
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class AblationConfig:
    """消融实验配置"""
    name: str
    description: str
    flags: List[str]
    is_baseline: bool = False


class AblationRunner:
    """消融实验运行器"""
    
    def __init__(self, base_command: List[str]):
        self.base_command = base_command
        self.results = {}
    
    def run_experiment(
        self,
        config: AblationConfig,
        extra_args: Dict[str, str] = None,
        timeout: int = 7200
    ) -> Tuple[float, str]:
        """
        运行单个消融实验
        
        Returns:
            accuracy: 准确率
            output: 输出日志
        """
        cmd = self.base_command.copy()
        
        # 添加消融标志
        for flag in config.flags:
            if '=' in flag:
                cmd.append(flag)
            else:
                cmd.append(flag)
        
        # 添加额外参数
        if extra_args:
            for k, v in extra_args.items():
                cmd.extend([f'--{k}', str(v)])
        
        print(f"\n{'='*60}")
        print(f"运行: {config.name}")
        print(f"命令: {' '.join(cmd)}")
        print(f"{'='*60}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=os.environ.copy()
            )
            
            output = result.stdout + result.stderr
            
            # 解析准确率
            accuracy = self._parse_accuracy(output)
            
            return accuracy, output
            
        except subprocess.TimeoutExpired:
            print(f"  ⏰ 超时 ({timeout}s)")
            return 0.0, "Timeout"
        except Exception as e:
            print(f"  ❌ 错误: {e}")
            return 0.0, str(e)
    
    def _parse_accuracy(self, output: str) -> float:
        """从输出中解析准确率"""
        import re
        
        # 尝试多种模式
        patterns = [
            r'Accuracy.*?(\d+\.?\d*)\%',
            r'准确率.*?(\d+\.?\d*)\%',
            r'Best.*?(\d+\.?\d*)\%',
            r'Val Acc.*?(\d+\.?\d*)\%',
            r'val_acc.*?(\d+\.?\d*)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                return float(match.group(1))
        
        return 0.0
    
    def run_all_ablations(
        self,
        configs: List[AblationConfig],
        extra_args: Dict[str, str] = None
    ) -> Dict[str, Dict]:
        """
        运行所有消融实验
        """
        results = {}
        
        for config in configs:
            acc, output = self.run_experiment(config, extra_args)
            
            results[config.name] = {
                'accuracy': acc,
                'description': config.description,
                'flags': config.flags,
                'is_baseline': config.is_baseline,
                'output': output[:500]  # 截断输出
            }
            
            print(f"\n  📊 {config.name}: {acc:.2f}%")
        
        return results
    
    def generate_report(self, results: Dict[str, Dict]) -> str:
        """生成消融实验报告"""
        report = []
        report.append("\n" + "="*60)
        report.append("消融实验结果报告")
        report.append("="*60)
        
        # 按基线排序
        baseline_name = None
        for name, data in results.items():
            if data.get('is_baseline', False):
                baseline_name = name
                break
        
        if baseline_name:
            baseline_acc = results[baseline_name]['accuracy']
            report.append(f"\n基线 ({baseline_name}): {baseline_acc:.2f}%")
            report.append("\n配置对比:")
            
            for name, data in results.items():
                if name == baseline_name:
                    continue
                acc = data['accuracy']
                diff = acc - baseline_acc
                sign = "+" if diff > 0 else ""
                report.append(f"  {name}: {acc:.2f}% ({sign}{diff:.2f}%)")
        
        report.append("\n" + "="*60)
        
        # 生成表格（论文格式）
        report.append("\n| 配置 | 准确率 | 相对基线 |")
        report.append("|:---|:---|:---|")
        
        for name, data in results.items():
            acc = data['accuracy']
            if name == baseline_name:
                report.append(f"| **{name}** | **{acc:.1f}%** | - |")
            else:
                diff = acc - results[baseline_name]['accuracy']
                report.append(f"| {name} | {acc:.1f}% | {diff:+.1f}% |")
        
        return "\n".join(report)


def get_paper_ablations() -> List[AblationConfig]:
    """
    论文 4.4 节的消融实验配置
    """
    return [
        AblationConfig(
            name="BERT baseline (无图结构)",
            description="仅使用BERT编码器，无图结构",
            flags=["--no_hypergraph"],
            is_baseline=True
        ),
        AblationConfig(
            name="+ 普通图 (PlotTree结构)",
            description="使用普通图结构替代超图",
            flags=["--use_graph", "--no_hyperedge"],
            is_baseline=False
        ),
        AblationConfig(
            name="+ 超图结构 (无类型感知)",
            description="使用超图但禁用类型感知",
            flags=["--no_type_aware"],
            is_baseline=False
        ),
        AblationConfig(
            name="+ 类型感知HGNE",
            description="启用类型感知超图卷积",
            flags=["--type_aware"],
            is_baseline=False
        ),
        AblationConfig(
            name="+ 超边感知检索HAR",
            description="启用超边感知检索",
            flags=["--use_har"],
            is_baseline=False
        ),
        AblationConfig(
            name="+ 预训练+微调 (完整)",
            description="完整模型",
            flags=["--use_pretrain", "--use_finetune", "--use_har", "--type_aware"],
            is_baseline=False
        )
    ]


def main():
    parser = argparse.ArgumentParser(description='消融实验自动化运行器')
    
    parser.add_argument('--script', default='experiments/har/run_qa.py', help='主脚本路径')
    parser.add_argument('--questions', required=True, help='问题文件')
    parser.add_argument('--hypergraph_dir', required=True, help='超图目录')
    parser.add_argument('--checkpoint', required=True, help='模型检查点')
    parser.add_argument('--output_dir', default='experiments/ablation_results')
    parser.add_argument('--dataset', default='storyvideoqa_g', help='数据集名称')
    parser.add_argument('--dry_run', action='store_true', help='仅打印命令不执行')
    
    args = parser.parse_args()
    
    # 构建基础命令
    base_cmd = [
        'python', args.script,
        '--questions', args.questions,
        '--hypergraph_dir', args.hypergraph_dir,
        '--checkpoint', args.checkpoint,
        '--output', os.path.join(args.output_dir, '${name}_result.json'),
        '--model_type', 'e2e'
    ]
    
    # 额外参数
    extra_args = {
        'dataset': args.dataset
    }
    
    if args.dry_run:
        print("Dry run mode - 仅打印命令")
        for config in get_paper_ablations():
            cmd = base_cmd.copy()
            cmd = [c.replace('${name}', config.name.replace(' ', '_')) for c in cmd]
            for flag in config.flags:
                cmd.append(flag)
            print(f"\n{config.name}: {' '.join(cmd)}")
        return
    
    runner = AblationRunner(base_cmd)
    results = runner.run_all_ablations(get_paper_ablations(), extra_args)
    
    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'ablation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # 生成报告
    report = runner.generate_report(results)
    print(report)
    
    with open(os.path.join(args.output_dir, 'ablation_report.txt'), 'w') as f:
        f.write(report)


if __name__ == '__main__':
    main()