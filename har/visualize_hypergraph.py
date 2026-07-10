"""
visualize_hypergraph.py - 超边可视化工具
用于论文附录 C.1 的超边可视化
"""

import os
import json
import argparse
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


class HypergraphVisualizer:
    """超图可视化器"""
    
    def __init__(self, hypergraph: Dict):
        self.hypergraph = hypergraph
        self.nodes = hypergraph.get('nodes', [])
        self.hyperedges = hypergraph.get('hyperedges', {})
        
        # 统一超边格式
        if isinstance(self.hyperedges, list):
            typed = {'MO': [], 'OM': [], 'CO': []}
            for e in self.hyperedges:
                tau = e.get('type', 'CO')
                typed[tau].append(e)
            self.hyperedges = typed
        
        self.node_id_to_idx = {n['id']: i for i, n in enumerate(self.nodes)}
        self.node_idx_to_id = {i: n['id'] for i, n in enumerate(self.nodes)}
    
    def _get_edge_color(self, edge_type: str) -> str:
        """获取超边类型对应的颜色"""
        colors = {
            'MO': '#FF6B6B',  # 红色 - 多因
            'OM': '#4ECDC4',  # 青色 - 多果
            'CO': '#45B7D1'   # 蓝色 - 共现
        }
        return colors.get(edge_type, '#888888')
    
    def _get_edge_label(self, edge_type: str) -> str:
        """获取超边类型标签"""
        labels = {
            'MO': '多因超边 (MO)',
            'OM': '多果超边 (OM)',
            'CO': '角色共现超边 (CO)'
        }
        return labels.get(edge_type, edge_type)
    
    def _calculate_node_positions(self, edge_type: str, edge_index: int) -> Dict:
        """计算节点布局"""
        if edge_type not in self.hyperedges:
            return {}
        
        edges = self.hyperedges[edge_type]
        if edge_index >= len(edges):
            return {}
        
        edge = edges[edge_index]
        node_ids = edge.get('nodes', [])
        
        # 只取前8个节点避免过于拥挤
        node_ids = node_ids[:8]
        
        if len(node_ids) < 2:
            return {}
        
        # 圆形布局
        num_nodes = len(node_ids)
        positions = {}
        
        for i, nid in enumerate(node_ids):
            angle = 2 * np.pi * i / num_nodes - np.pi / 2
            x = 0.5 + 0.4 * np.cos(angle)
            y = 0.5 + 0.4 * np.sin(angle)
            positions[nid] = (x, y)
        
        # 特殊处理：MO超边 - 中心节点为果，周围为因
        if edge_type == 'MO' and len(edge.get('source_nodes', [])) > 1:
            # 果节点在中心
            target_nodes = edge.get('target_nodes', [])
            source_nodes = edge.get('source_nodes', [])
            
            if target_nodes and source_nodes:
                # 果节点在中心
                positions[target_nodes[0]] = (0.5, 0.5)
                
                # 因节点在周围
                num_sources = len(source_nodes)
                for i, nid in enumerate(source_nodes[:7]):
                    angle = 2 * np.pi * i / num_sources
                    x = 0.5 + 0.4 * np.cos(angle)
                    y = 0.5 + 0.4 * np.sin(angle)
                    positions[nid] = (x, y)
        
        # 特殊处理：OM超边 - 中心节点为因，周围为果
        elif edge_type == 'OM' and len(edge.get('target_nodes', [])) > 1:
            source_nodes = edge.get('source_nodes', [])
            target_nodes = edge.get('target_nodes', [])
            
            if source_nodes and target_nodes:
                # 因节点在中心
                positions[source_nodes[0]] = (0.5, 0.5)
                
                # 果节点在周围
                for i, nid in enumerate(target_nodes[:7]):
                    angle = 2 * np.pi * i / len(target_nodes[:7])
                    x = 0.5 + 0.4 * np.cos(angle)
                    y = 0.5 + 0.4 * np.sin(angle)
                    positions[nid] = (x, y)
        
        return positions
    
    def _get_node_label(self, node_id: str) -> str:
        """获取节点标签"""
        for n in self.nodes:
            if n['id'] == node_id:
                # 取摘要或动作的前15个字符
                text = n.get('S_i', n.get('A_i', n.get('text', '')))
                if text:
                    return text[:15] + '...' if len(text) > 15 else text
                return node_id
        return node_id
    
    def _get_node_color(self, node_id: str) -> str:
        """获取节点颜色"""
        for n in self.nodes:
            if n['id'] == node_id:
                # 根据是否有角色信息着色
                if n.get('P_i'):
                    return '#FFD93D'  # 黄色 - 有角色
                return '#6BCB77'  # 绿色 - 无角色
        return '#888888'
    
    def visualize_hyperedge(
        self,
        edge_type: str,
        edge_index: int,
        output_path: str,
        figsize: Tuple[int, int] = (8, 6)
    ):
        """可视化单个超边"""
        if edge_type not in self.hyperedges:
            print(f"Error: No {edge_type} edges")
            return
        
        edges = self.hyperedges[edge_type]
        if edge_index >= len(edges):
            print(f"Error: Edge index {edge_index} out of range (max {len(edges)-1})")
            return
        
        edge = edges[edge_index]
        node_ids = edge.get('nodes', [])[:8]
        
        if len(node_ids) < 2:
            print("Error: Edge has less than 2 nodes")
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 计算位置
        positions = self._calculate_node_positions(edge_type, edge_index)
        
        # 绘制超边（阴影区域）
        if len(node_ids) >= 3:
            edge_coords = [positions[nid] for nid in node_ids if nid in positions]
            if len(edge_coords) >= 3:
                from scipy.spatial import ConvexHull
                try:
                    hull = ConvexHull(edge_coords)
                    hull_points = [edge_coords[i] for i in hull.vertices]
                    hull_points.append(hull_points[0])
                    
                    patch = patches.Polygon(
                        hull_points,
                        closed=True,
                        alpha=0.15,
                        color=self._get_edge_color(edge_type),
                        edgecolor=self._get_edge_color(edge_type),
                        linewidth=0.5,
                        linestyle='--'
                    )
                    ax.add_patch(patch)
                except:
                    pass
        
        # 绘制节点
        node_colors = [self._get_node_color(nid) for nid in node_ids if nid in positions]
        
        for nid in node_ids:
            if nid not in positions:
                continue
            
            x, y = positions[nid]
            label = self._get_node_label(nid)
            color = self._get_node_color(nid)
            
            # 节点圆形
            circle = plt.Circle((x, y), 0.08, color=color, alpha=0.8)
            ax.add_patch(circle)
            
            # 节点标签
            ax.text(x, y - 0.12, label, ha='center', va='center', fontsize=8, wrap=True)
        
        # 绘制连线（节点间的关系）
        for i, nid1 in enumerate(node_ids):
            if nid1 not in positions:
                continue
            for j, nid2 in enumerate(node_ids):
                if i >= j or nid2 not in positions:
                    continue
                x1, y1 = positions[nid1]
                x2, y2 = positions[nid2]
                
                # 添加箭头（表示因果关系）
                if edge_type in ['MO', 'OM']:
                    # MO: 因→果, OM: 因←果
                    if edge_type == 'MO':
                        # 检查谁是果
                        target_nodes = edge.get('target_nodes', [])
                        if nid2 in target_nodes:
                            dx, dy = x2 - x1, y2 - y1
                            ax.arrow(x1, y1, dx*0.8, dy*0.8, 
                                    head_width=0.03, head_length=0.03,
                                    fc=self._get_edge_color(edge_type),
                                    ec=self._get_edge_color(edge_type),
                                    alpha=0.5, length_includes_head=True)
                    else:
                        source_nodes = edge.get('source_nodes', [])
                        if nid1 in source_nodes:
                            dx, dy = x2 - x1, y2 - y1
                            ax.arrow(x1, y1, dx*0.8, dy*0.8,
                                    head_width=0.03, head_length=0.03,
                                    fc=self._get_edge_color(edge_type),
                                    ec=self._get_edge_color(edge_type),
                                    alpha=0.5, length_includes_head=True)
                else:
                    # CO: 无方向
                    ax.plot([x1, x2], [y1, y2], color='#888888', alpha=0.3, linestyle=':')
        
        # 设置图形
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # 标题
        title = f"{self._get_edge_label(edge_type)}\n"
        title += f"节点数: {len(node_ids)}"
        if edge_type in ['MO', 'OM']:
            if edge.get('source_nodes'):
                title += f" | 因: {len(edge['source_nodes'])}"
            if edge.get('target_nodes'):
                title += f" | 果: {len(edge['target_nodes'])}"
        title += f"\n权重: {edge.get('weight', 1.0):.2f}"
        ax.set_title(title, fontsize=12, pad=20)
        
        # 图例
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', label='事件节点',
                      markerfacecolor='#FFD93D', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='含角色信息',
                      markerfacecolor='#6BCB77', markersize=10)
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 可视化保存至: {output_path}")
    
    def visualize_hypergraph_summary(
        self,
        output_path: str,
        max_edges_per_type: int = 3,
        figsize: Tuple[int, int] = (15, 10)
    ):
        """可视化超图摘要"""
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        for ax_idx, edge_type in enumerate(['MO', 'OM', 'CO']):
            ax = axes[ax_idx]
            edges = self.hyperedges.get(edge_type, [])
            
            if not edges:
                ax.text(0.5, 0.5, f'无 {edge_type} 超边', 
                       ha='center', va='center', fontsize=12)
                ax.axis('off')
                continue
            
            # 选择最大的几个超边
            sorted_edges = sorted(edges, key=lambda e: len(e.get('nodes', [])), reverse=True)
            selected_edges = sorted_edges[:max_edges_per_type]
            
            # 绘制每个超边
            for e_idx, edge in enumerate(selected_edges):
                node_ids = edge.get('nodes', [])[:6]
                if len(node_ids) < 2:
                    continue
                
                # 计算位置
                num_nodes = len(node_ids)
                positions = {}
                for i, nid in enumerate(node_ids):
                    angle = 2 * np.pi * i / num_nodes + e_idx * 0.3
                    x = 0.5 + 0.35 * np.cos(angle)
                    y = 0.5 + 0.35 * np.sin(angle)
                    positions[nid] = (x, y)
                
                # 绘制超边区域
                if len(node_ids) >= 3:
                    coords = [positions[nid] for nid in node_ids if nid in positions]
                    if len(coords) >= 3:
                        from scipy.spatial import ConvexHull
                        try:
                            hull = ConvexHull(coords)
                            hull_points = [coords[i] for i in hull.vertices]
                            hull_points.append(hull_points[0])
                            patch = patches.Polygon(
                                hull_points, closed=True,
                                alpha=0.1 + 0.1 * e_idx,
                                color=self._get_edge_color(edge_type),
                                edgecolor=self._get_edge_color(edge_type),
                                linewidth=0.5
                            )
                            ax.add_patch(patch)
                        except:
                            pass
                
                # 绘制节点
                for nid in node_ids:
                    if nid not in positions:
                        continue
                    x, y = positions[nid]
                    ax.plot(x, y, 'o', markersize=8, 
                           color=self._get_node_color(nid), alpha=0.8)
                    ax.text(x, y - 0.05, nid[:6], ha='center', va='center', fontsize=6)
            
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.set_aspect('equal')
            ax.axis('off')
            ax.set_title(f'{self._get_edge_label(edge_type)}\n({len(edges)} 条超边)')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 摘要可视化保存至: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='超图可视化工具')
    
    parser.add_argument('--hypergraph', required=True, help='超图JSON文件')
    parser.add_argument('--output_dir', default='experiments/visualizations')
    parser.add_argument('--edge_type', choices=['MO', 'OM', 'CO', 'all'], default='all')
    parser.add_argument('--edge_index', type=int, default=0)
    parser.add_argument('--max_edges', type=int, default=3)
    
    args = parser.parse_args()
    
    # 加载超图
    with open(args.hypergraph, 'r', encoding='utf-8') as f:
        hypergraph = json.load(f)
    
    visualizer = HypergraphVisualizer(hypergraph)
    os.makedirs(args.output_dir, exist_ok=True)
    
    video_id = hypergraph.get('video_id', 'unknown')
    
    if args.edge_type == 'all':
        # 生成摘要
        output_path = os.path.join(args.output_dir, f'{video_id}_summary.png')
        visualizer.visualize_hypergraph_summary(
            output_path, max_edges_per_type=args.max_edges
        )
        
        # 为每种类型生成示例
        for edge_type in ['MO', 'OM', 'CO']:
            if edge_type in visualizer.hyperedges:
                edges = visualizer.hyperedges[edge_type]
                if edges:
                    output_path = os.path.join(args.output_dir, f'{video_id}_{edge_type}_example.png')
                    visualizer.visualize_hyperedge(
                        edge_type, 0, output_path
                    )
    else:
        output_path = os.path.join(args.output_dir, f'{video_id}_{args.edge_type}.png')
        visualizer.visualize_hyperedge(
            args.edge_type, args.edge_index, output_path
        )


if __name__ == '__main__':
    main()