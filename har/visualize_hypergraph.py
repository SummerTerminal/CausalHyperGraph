"""
visualize_hypergraph.py - hyperedge visualization (appendix C.1)
"""

import os, json, argparse
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


class HypergraphVisualizer:
    """Visualizer for causal hypergraphs."""

    def __init__(self, hg: Dict):
        self.hg = hg
        self.nds = hg.get('nodes', [])
        self.hedges = hg.get('hyperedges', {})
        # 统一成 dict of list
        if isinstance(self.hedges, list):
            typed = {'MO': [], 'OM': [], 'CO': []}
            for e in self.hedges:
                tau = e.get('type', 'CO')
                typed[tau].append(e)
            self.hedges = typed

        self.nid2idx = {n['id']: i for i, n in enumerate(self.nds)}
        self.idx2nid = {i: n['id'] for i, n in enumerate(self.nds)}

    def _edge_color(self, etype: str) -> str:
        # color per hyperedge type
        cols = {
            'MO': '#FF6B6B',  # red - multi-cause
            'OM': '#4ECDC4',  # teal - multi-effect
            'CO': '#45B7D1'  # blue - co-occurrence
        }
        return cols.get(etype, '#888888')

    def _edge_label(self, etype: str) -> str:
        labs = {
            'MO': 'MO (multi-cause)',
            'OM': 'OM (multi-effect)',
            'CO': 'CO (character co-occurrence)'
        }
        return labs.get(etype, etype)

    def _node_positions(self, etype: str, eidx: int) -> Dict:
        # compute layout for one hyperedge
        if etype not in self.hedges:
            return {}
        edges = self.hedges[etype]
        if eidx >= len(edges):
            return {}
        edge = edges[eidx]
        nids = edge.get('nodes', [])[:8]  # avoid clutter
        if len(nids) < 2:
            return {}

        N = len(nids)
        pos = {}
        for i, nid in enumerate(nids):
            ang = 2 * np.pi * i / N - np.pi / 2
            pos[nid] = (0.5 + 0.4 * np.cos(ang), 0.5 + 0.4 * np.sin(ang))

        # special layout for MO: central effect node
        if etype == 'MO' and len(edge.get('source_nodes', [])) > 1:
            tgt = edge.get('target_nodes', [])
            src = edge.get('source_nodes', [])
            if tgt and src:
                pos[tgt[0]] = (0.5, 0.5)
                for i, nid in enumerate(src[:7]):
                    ang = 2 * np.pi * i / len(src[:7])
                    pos[nid] = (0.5 + 0.4 * np.cos(ang), 0.5 + 0.4 * np.sin(ang))

        # special layout for OM: central cause node
        elif etype == 'OM' and len(edge.get('target_nodes', [])) > 1:
            src = edge.get('source_nodes', [])
            tgt = edge.get('target_nodes', [])
            if src and tgt:
                pos[src[0]] = (0.5, 0.5)
                for i, nid in enumerate(tgt[:7]):
                    ang = 2 * np.pi * i / len(tgt[:7])
                    pos[nid] = (0.5 + 0.4 * np.cos(ang), 0.5 + 0.4 * np.sin(ang))

        return pos

    def _node_text(self, nid: str) -> str:
        for n in self.nds:
            if n['id'] == nid:
                txt = n.get('S_i', n.get('A_i', n.get('text', '')))
                if txt:
                    return txt[:15] + ('...' if len(txt) > 15 else '')
                return nid
        return nid

    def _node_color(self, nid: str) -> str:
        for n in self.nds:
            if n['id'] == nid:
                return '#FFD93D' if n.get('P_i') else '#6BCB77'  # yellow if has character
        return '#888888'

    def draw_single_edge(self, etype: str, eidx: int, out_path: str, figsize=(8, 6)):
        """visualize a single hyperedge"""
        if etype not in self.hedges:
            print(f"No {etype} edges")
            return
        edges = self.hedges[etype]
        if eidx >= len(edges):
            print(f"Edge idx {eidx} out of range (max {len(edges) - 1})")
            return

        edge = edges[eidx]
        nids = edge.get('nodes', [])[:8]
        if len(nids) < 2:
            print("Edge has <2 nodes")
            return

        fig, ax = plt.subplots(figsize=figsize)
        pos = self._node_positions(etype, eidx)

        # 画凸包区域 (convex hull for hyperedge area)
        if len(nids) >= 3:
            coords = [pos[n] for n in nids if n in pos]
            if len(coords) >= 3:
                try:
                    from scipy.spatial import ConvexHull
                    hull = ConvexHull(coords)
                    pts = [coords[i] for i in hull.vertices] + [coords[0]]
                    poly = patches.Polygon(pts, closed=True, alpha=0.15,
                                           color=self._edge_color(etype),
                                           edgecolor=self._edge_color(etype),
                                           linewidth=0.5, linestyle='--')
                    ax.add_patch(poly)
                except:
                    pass

        # draw nodes
        for nid in nids:
            if nid not in pos: continue
            x, y = pos[nid]
            ax.add_patch(plt.Circle((x, y), 0.08, color=self._node_color(nid), alpha=0.8))
            ax.text(x, y - 0.12, self._node_text(nid), ha='center', va='center', fontsize=8)

        # draw edges between nodes (causal arrows)
        for i, n1 in enumerate(nids):
            if n1 not in pos: continue
            for j, n2 in enumerate(nids):
                if i >= j or n2 not in pos: continue
                x1, y1 = pos[n1];
                x2, y2 = pos[n2]
                if etype in ('MO', 'OM'):
                    if etype == 'MO':
                        tgt_nodes = edge.get('target_nodes', [])
                        if n2 in tgt_nodes:
                            dx, dy = x2 - x1, y2 - y1
                            ax.arrow(x1, y1, dx * 0.8, dy * 0.8, head_width=0.03, head_length=0.03,
                                     fc=self._edge_color(etype), ec=self._edge_color(etype),
                                     alpha=0.5, length_includes_head=True)
                    else:  # OM
                        src_nodes = edge.get('source_nodes', [])
                        if n1 in src_nodes:
                            dx, dy = x2 - x1, y2 - y1
                            ax.arrow(x1, y1, dx * 0.8, dy * 0.8, head_width=0.03, head_length=0.03,
                                     fc=self._edge_color(etype), ec=self._edge_color(etype),
                                     alpha=0.5, length_includes_head=True)
                else:
                    ax.plot([x1, x2], [y1, y2], color='#888888', alpha=0.3, linestyle=':')

        ax.set_xlim(-0.05, 1.05);
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal');
        ax.axis('off')

        title = f"{self._edge_label(etype)}\nNodes: {len(nids)}"
        if etype in ('MO', 'OM'):
            if edge.get('source_nodes'): title += f" | causes: {len(edge['source_nodes'])}"
            if edge.get('target_nodes'): title += f" | effects: {len(edge['target_nodes'])}"
        title += f"\nweight: {edge.get('weight', 1.0):.2f}"
        ax.set_title(title, fontsize=12, pad=20)

        # legend
        ax.legend(handles=[
            plt.Line2D([0], [0], marker='o', color='w', label='event node', markerfacecolor='#FFD93D', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='has character', markerfacecolor='#6BCB77', markersize=10)
        ], loc='upper right', fontsize=8)

        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ saved to {out_path}")

    def draw_summary(self, out_path: str, max_per_type=3, figsize=(15, 10)):
        """visualize hypergraph overview"""
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        for axi, etype in enumerate(['MO', 'OM', 'CO']):
            ax = axes[axi]
            edges = self.hedges.get(etype, [])
            if not edges:
                ax.text(0.5, 0.5, f'no {etype} edges', ha='center', va='center', fontsize=12)
                ax.axis('off')
                continue
            # 挑最大的几个超边
            sorted_edges = sorted(edges, key=lambda e: len(e.get('nodes', [])), reverse=True)
            selected = sorted_edges[:max_per_type]
            for eidx, edge in enumerate(selected):
                nids = edge.get('nodes', [])[:6]
                if len(nids) < 2: continue
                pos = {}
                for i, nid in enumerate(nids):
                    ang = 2 * np.pi * i / len(nids) + eidx * 0.3
                    pos[nid] = (0.5 + 0.35 * np.cos(ang), 0.5 + 0.35 * np.sin(ang))
                if len(nids) >= 3:
                    coords = [pos[n] for n in nids if n in pos]
                    if len(coords) >= 3:
                        try:
                            from scipy.spatial import ConvexHull
                            hull = ConvexHull(coords)
                            pts = [coords[i] for i in hull.vertices] + [coords[0]]
                            poly = patches.Polygon(pts, closed=True, alpha=0.1 + 0.1 * eidx,
                                                   color=self._edge_color(etype),
                                                   edgecolor=self._edge_color(etype), linewidth=0.5)
                            ax.add_patch(poly)
                        except:
                            pass
                for nid in nids:
                    if nid not in pos: continue
                    x, y = pos[nid]
                    ax.plot(x, y, 'o', markersize=8, color=self._node_color(nid), alpha=0.8)
                    ax.text(x, y - 0.05, nid[:6], ha='center', va='center', fontsize=6)
            ax.set_xlim(-0.05, 1.05);
            ax.set_ylim(-0.05, 1.05)
            ax.set_aspect('equal');
            ax.axis('off')
            ax.set_title(f'{self._edge_label(etype)}\n({len(edges)} edges)')

        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ summary saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description='hypergraph visualizer')
    parser.add_argument('--hypergraph', required=True)
    parser.add_argument('--output_dir', default='experiments/visualizations')
    parser.add_argument('--edge_type', choices=['MO', 'OM', 'CO', 'all'], default='all')
    parser.add_argument('--edge_index', type=int, default=0)
    parser.add_argument('--max_edges', type=int, default=3)
    args = parser.parse_args()

    with open(args.hypergraph, 'r', encoding='utf-8') as f:
        hg = json.load(f)

    vis = HypergraphVisualizer(hg)
    os.makedirs(args.output_dir, exist_ok=True)
    vid = hg.get('video_id', 'unknown')

    if args.edge_type == 'all':
        vis.draw_summary(os.path.join(args.output_dir, f'{vid}_summary.png'),
                         max_per_type=args.max_edges)
        for et in ['MO', 'OM', 'CO']:
            if et in vis.hedges and vis.hedges[et]:
                vis.draw_single_edge(et, 0, os.path.join(args.output_dir, f'{vid}_{et}_example.png'))
    else:
        vis.draw_single_edge(args.edge_type, args.edge_index,
                             os.path.join(args.output_dir, f'{vid}_{args.edge_type}.png'))


if __name__ == '__main__':
    main()