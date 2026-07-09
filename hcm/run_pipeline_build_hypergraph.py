"""
run_pipeline_build_hypergraph.py - 数据处理与并行调度 Pipeline

职责：
1. 读取 aligned_script 目录下的 Excel 文件
2. 提取场景/事件数据（整合 script + subtitle 工作表）
3. 生成 BERT embeddings（GPU）
4. 调用 build_hypergraph.py 构建超图
5. CPU-GPU 流水线并行调度

支持数据格式：
- BigBang/Friends: 5个工作表（subtitle_*, script_*, results_*, results_human, scripts）
- GOT/Movie: 4个工作表（subtitle, script, results, results_human）

# 严格模式 + 密度控制（推荐）
python experiments/hcm/run_pipeline_build_hypergraph.py \
    --aligned_script_dir files/baseline/StoryVideoQA-main/StoryMindv2/aligned_script/GOT \
    --output_dir experiments/hcm/hypergraphs \
    --batch_size 32 \
    --max_parallel 6 \
    --cache_dir ckpt \
    --strict \
    --mo_size 3 \
    --om_size 3 \
    --max_mo_ratio 0.3 \
    --max_om_ratio 0.3 \
    --enable_debug

# 如果需要对比宽松模式
python experiments/hcm/run_pipeline_build_hypergraph.py \
    --aligned_script_dir files/baseline/StoryVideoQA-main/StoryMindv2/aligned_script/GOT \
    --output_dir experiments/hcm/hypergraphs_loose \
    --batch_size 32 \
    --max_parallel 6 \
    --cache_dir ckpt \
    --no_strict \
    --mo_size 5 \
    --om_size 5

"""
import os
import json
import argparse
import time
import pandas as pd
import torch
import numpy as np

from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import glob
from typing import List, Dict, Tuple, Optional
import re

from build_hypergraph import build_hypergraph

import sys
import io
# 修复 Windows Git Bash 编码问题
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except:
        pass

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


# ============================================================
# 第一部分：Excel 数据解析
# ============================================================

def parse_time_to_seconds(time_str: str) -> Optional[float]:
    """将时间字符串转换为秒"""
    if not time_str or not isinstance(time_str, str):
        return None
    
    time_str = time_str.strip()
    if not time_str:
        return None
    
    time_str = time_str.replace(',', '.')
    
    match = re.match(r'(\d+):(\d+):(\d+)(?:\.(\d+))?', time_str)
    if match:
        h, m, s = int(match.group(1)), int(match.group(2)), int(match.group(3))
        ms = match.group(4)
        if ms:
            ms = ms.ljust(3, '0')[:3]
            s += int(ms) / 1000.0
        return h * 3600 + m * 60 + s
    
    return None

def is_valid_string(s: str) -> bool:
    """检查字符串是否有效"""
    if not s or not isinstance(s, str):
        return False
    s_stripped = s.strip()
    if not s_stripped:
        return False
    invalid_values = {'nan', 'none', 'null', 'na', '-', '--', 'n/a', ''}
    if s_stripped.lower() in invalid_values:
        return False
    return True


def detect_sheet_format(xl: pd.ExcelFile) -> Dict[str, str]:
    """检测 Excel 工作表格式"""
    sheets = xl.sheet_names
    subtitle_sheet = None
    script_sheet = None
    
    for s in sheets:
        if s.startswith('subtitle_'):
            subtitle_sheet = s
        elif s.startswith('script_') and not s.startswith('scripts'):
            script_sheet = s
    
    if subtitle_sheet is None and 'subtitle' in sheets:
        subtitle_sheet = 'subtitle'
    if script_sheet is None:
        if 'scripts' in sheets:
            script_sheet = 'scripts'
        elif 'script' in sheets:
            script_sheet = 'script'
    
    return {
        'subtitle_sheet': subtitle_sheet,
        'script_sheet': script_sheet
    }


def generate_summary(content: str, action: str, characters: List[str]) -> str:
    """
    生成事件摘要
    
    方案B：改进规则摘要（无LLM时）
    预留LLM接口，可通过环境变量启用
    """
    # 检查是否启用LLM
    use_llm = os.getenv('USE_LLM_SUMMARY', 'false').lower() == 'true'
    
    if use_llm:
        try:
            import google.generativeai as genai
            api_key = os.getenv('GEMINI_API_KEY')
            if api_key:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-2.0-flash')
                prompt = f"Summarize this movie scene in one sentence (max 50 words): {content[:1000]}"
                response = model.generate_content(prompt)
                return response.text.strip()
        except Exception as e:
            print(f"  LLM summary failed: {e}, fallback to heuristic")
    
    # 改进规则摘要：提取首句 + 角色信息
    first_sentence = content.split('.')[0] if '.' in content else content
    summary = first_sentence[:250]
    if characters:
        summary = f"{' '.join(characters[:3])}: {summary}"
    return summary


def extract_scenes_from_excel(xlsx_path: str) -> List[Dict]:
    """
    从 Excel 文件中提取场景/事件列表
    
    核心修复：使用 results 工作表的 index_result 列
    将字幕时间戳精确映射到 script 工作表的对应行
    """
    xl = pd.ExcelFile(xlsx_path)
    sheet_info = detect_sheet_format(xl)
    
    if not sheet_info['script_sheet']:
        print(f"  ⚠️  No script sheet found in {xlsx_path}")
        return []
    
    script_df = pd.read_excel(xl, sheet_name=sheet_info['script_sheet'])
    
    # Step 2: 找到对齐结果表
    results_sheet = None
    for s in xl.sheet_names:
        if s.startswith('results_') and 'human' not in s:
            results_sheet = s
            break
    if not results_sheet:
        for s in xl.sheet_names:
            if s == 'results':
                results_sheet = s
                break
    if not results_sheet:
        for s in xl.sheet_names:
            if 'results_human' in s:
                results_sheet = s
                print(f"  ⚠️  Using human-annotated alignment: {results_sheet}")
                break
    
    time_map = {}
    if results_sheet:
        try:
            results_df = pd.read_excel(xl, sheet_name=results_sheet)
            for _, row in results_df.iterrows():
                idx = row.get('index_result')
                if idx is None or pd.isna(idx):
                    continue
                idx = int(idx)
                if idx not in time_map:
                    time_map[idx] = {
                        'start_time': str(row.get('start_time', '')).strip(),
                        'end_time': str(row.get('end_time', '')).strip(),
                        'dialog': str(row.get('dialog', '')).strip()
                    }
                else:
                    time_map[idx]['end_time'] = str(row.get('end_time', '')).strip()
        except Exception as e:
            print(f"  ⚠️  Could not read results sheet '{results_sheet}': {e}")
    
    # Step 3: 遍历 script 表构建事件列表
    events = []
    current_scene_index = 1
    current_location = ""
    current_description = ""
    
    def _clean(s):
        s = str(s).strip()
        return '' if s.lower() in ['nan', 'none', ''] else s
    
    for idx, row in script_df.iterrows():
        if 'Unnamed: 0' in script_df.columns:
            excel_row_num = int(row['Unnamed: 0'])
        else:
            excel_row_num = idx + 1
        
        record_type = str(row.get('record_type', '')).strip().lower()
        characters = _clean(str(row.get('characters', '')))
        location = _clean(str(row.get('location', '')))
        content = _clean(str(row.get('content', '')))
        description = _clean(str(row.get('description', '')))
        
        if record_type == 'scene':
            if 'scene_index' in script_df.columns:
                si = row.get('scene_index')
                if si and not pd.isna(si):
                    current_scene_index = int(si)
            if location:
                current_location = location
            if description:
                current_description = description
            continue
        
        if record_type in ['dialog', ''] or not record_type:
            if not content:
                if description:
                    content = description
                else:
                    continue
            
            # 修复：使用 is_valid_string 统一过滤，保留单字符角色名
            if characters:
                char_list = [c.strip() for c in re.split(r'[,/&]', characters) 
                           if is_valid_string(c.strip())]
            else:
                char_list = []
            
            timestamp = time_map.get(excel_row_num, {})
            start_time = timestamp.get('start_time', '')
            end_time = timestamp.get('end_time', '')
            time_seconds = parse_time_to_seconds(start_time)
            
            text_parts = []
            if current_location:
                text_parts.append(f"Location: {current_location}.")
            if char_list:
                text_parts.append(f"Characters: {', '.join(char_list)}.")
            if content:
                text_parts.append(f"Dialog: {content}")
            if description:
                text_parts.append(f"Description: {description}")
            
            full_text = ' '.join(text_parts)
            
            # 修复：使用改进的摘要生成
            summary = generate_summary(content, content[:200], char_list)
            
            events.append({
                'id': '',
                'time_seconds': time_seconds,
                'start_time': start_time,
                'end_time': end_time,
                'P_i': char_list,
                'A_i': content[:200] if len(content) > 200 else content,
                'L_i': current_location,
                'S_i': summary,
                'text': full_text,
                'description': description,
                'scene_index': current_scene_index,
            })
    
    for i, event in enumerate(events):
        event['id'] = f'e_{i:04d}'
    
    events_with_time = sum(1 for e in events if e['time_seconds'] is not None)
    print(f"  📊 Extracted {len(events)} events, "
          f"{events_with_time}/{len(events)} ({100*events_with_time/max(1,len(events)):.1f}%) with timestamps")
    
    return events


# ============================================================
# 第二部分：BERT Embedding 编码器
# ============================================================

class EmbeddingEncoder:
    """BERT 编码器（GPU 加速）"""
    
    def __init__(self, cache_dir: str = 'ckpt', batch_size: int = 32):
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading bert-large-uncased on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased', cache_dir=cache_dir)
        self.model = AutoModel.from_pretrained('bert-large-uncased', cache_dir=cache_dir)
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}")
    
    @torch.no_grad()
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """批量编码文本，返回 [batch_size, 1024] embeddings"""
        inputs = self.tokenizer(
            texts, 
            return_tensors='pt', 
            padding=True, 
            truncation=True, 
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return embeddings
    
    def encode_events(self, events: List[Dict]) -> List[Dict]:
        """为事件列表生成 embeddings"""
        texts = []
        for e in events:
            t = e.get('text', '')
            if not t or not t.strip():
                t = e.get('S_i', 'No text')
            texts.append(t)
        
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = self.encode_batch(batch_texts)
            all_embeddings.append(batch_embeddings)
        
        if all_embeddings:
            all_embeddings = np.concatenate(all_embeddings, axis=0)
        else:
            all_embeddings = np.array([])
        
        for i, e in enumerate(events):
            if i < len(all_embeddings):
                e['embedding'] = all_embeddings[i].tolist()
            else:
                e['embedding'] = []
        
        return events


# ============================================================
# 第三部分：并行调度 Pipeline
# ============================================================

class HypergraphBuildPipeline:
    """超图构建流水线（CPU预处理 + GPU编码 + 超图构建）"""
    
    def __init__(self, cache_dir: str = 'ckpt', batch_size: int = 32):
        self.encoder = EmbeddingEncoder(cache_dir=cache_dir, batch_size=batch_size)
    
    def process_single_video(self, xlsx_path: str, output_path: str, 
                              video_id: str, video_lengths: Optional[Dict] = None,
                              **hypergraph_kwargs) -> Dict:
        """处理单个视频：提取事件 → 编码 → 构建超图 → 保存"""
        if os.path.exists(output_path):
            print(f"  ⏭️  {video_id}: already exists, skipping")
            return None
        
        t_start = time.time()
        
        events = extract_scenes_from_excel(xlsx_path)
        if not events:
            print(f"  ⚠️  {video_id}: no events extracted")
            return None
        
        events = self.encoder.encode_events(events)
        
        E_MO, E_OM, E_CO, stats = build_hypergraph(
            events=events,
            video_id=video_id,
            video_lengths=video_lengths,
            **hypergraph_kwargs
        )
        
        hypergraph = {
            'video_id': video_id,
            'nodes': [{
                'id': e['id'],
                'time_seconds': e.get('time_seconds'),
                'start_time': e.get('start_time', ''),
                'end_time': e.get('end_time', ''),
                'P_i': e.get('P_i', []),
                'A_i': e.get('A_i', ''),
                'L_i': e.get('L_i', ''),
                'S_i': e.get('S_i', ''),
                'text': e.get('text', ''),
                'embedding': e.get('embedding', [])
            } for e in events],
            'hyperedges': {
                'MO': E_MO,
                'OM': E_OM,
                'CO': E_CO
            },
            'stats': stats
        }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(hypergraph, f, ensure_ascii=False, indent=2)
        
        t_elapsed = time.time() - t_start
        print(f"  ✅ {video_id}: {stats['num_events']} events, "
              f"MO:{stats['num_mo_edges']} OM:{stats['num_om_edges']} CO:{stats['num_co_edges']} "
              f"({t_elapsed:.1f}s)")
        
        return stats
    
    def run_parallel(self, video_list: List[Tuple[str, str, str]], 
                     max_parallel: int = 4,
                     video_lengths: Optional[Dict] = None,
                     **hypergraph_kwargs) -> Dict:
        """并行处理多个视频"""
        total = len(video_list)
        results = []
        errors = []
        skipped = 0
        
        import concurrent.futures
        
        def preprocess_task(video_id, xlsx_path, output_path):
            if os.path.exists(output_path):
                return ('skip', video_id, None, None)
            try:
                events = extract_scenes_from_excel(xlsx_path)
                return ('ready', video_id, events, output_path)
            except Exception as e:
                return ('error', video_id, str(e), None)
        
        print(f"\n{'='*60}")
        print(f"Phase 1: Preprocessing {total} videos (max_parallel={max_parallel})")
        print(f"{'='*60}")
        
        preprocess_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as executor:
            futures = {
                executor.submit(preprocess_task, vid, xlsx, out): vid
                for vid, xlsx, out in video_list
            }
            for future in concurrent.futures.as_completed(futures):
                status, vid, data, out_path = future.result()
                if status == 'skip':
                    skipped += 1
                elif status == 'error':
                    errors.append(vid)
                    print(f"  ❌ {vid}: {data}")
                else:
                    preprocess_results.append((vid, data, out_path))
        
        print(f"\n{'='*60}")
        print(f"Phase 2: Encoding + Building hypergraphs ({len(preprocess_results)} videos)")
        print(f"{'='*60}")
        
        for i, (vid, events, out_path) in enumerate(preprocess_results):
            print(f"\n[{i+1}/{len(preprocess_results)}] Processing {vid}...")
            try:
                events = self.encoder.encode_events(events)
                E_MO, E_OM, E_CO, stats = build_hypergraph(
                    events=events,
                    video_id=vid,
                    video_lengths=video_lengths,
                    **hypergraph_kwargs
                )
                
                hypergraph = {
                    'video_id': vid,
                    'nodes': [{
                        'id': e['id'],
                        'time_seconds': e.get('time_seconds'),
                        'start_time': e.get('start_time', ''),
                        'end_time': e.get('end_time', ''),
                        'P_i': e.get('P_i', []),
                        'A_i': e.get('A_i', ''),
                        'L_i': e.get('L_i', ''),
                        'S_i': e.get('S_i', ''),
                        'text': e.get('text', ''),
                        'embedding': e.get('embedding', [])
                    } for e in events],
                    'hyperedges': {
                        'MO': E_MO,
                        'OM': E_OM,
                        'CO': E_CO
                    },
                    'stats': stats
                }
                
                Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(hypergraph, f, ensure_ascii=False, indent=2)
                
                results.append(stats)
                print(f"  ✅ {vid}: {stats['num_events']} events, "
                      f"MO:{stats['num_mo_edges']} OM:{stats['num_om_edges']} CO:{stats['num_co_edges']}")
                
            except Exception as e:
                errors.append(vid)
                print(f"  ❌ {vid}: {e}")
        
        summary = {
            'total_videos': total,
            'processed': len(results),
            'skipped': skipped,
            'errors': len(errors),
            'error_videos': errors,
            'total_events': sum(r['num_events'] for r in results),
            'total_mo_edges': sum(r['num_mo_edges'] for r in results),
            'total_om_edges': sum(r['num_om_edges'] for r in results),
            'total_co_edges': sum(r['num_co_edges'] for r in results),
        }
        
        print(f"\n{'='*60}")
        print(f"Pipeline Complete")
        print(f"{'='*60}")
        print(f"  Total: {total}, Processed: {len(results)}, Skipped: {skipped}, Errors: {len(errors)}")
        print(f"  Events: {summary['total_events']}")
        print(f"  MO: {summary['total_mo_edges']}, OM: {summary['total_om_edges']}, CO: {summary['total_co_edges']}")
        print(f"{'='*60}\n")
        
        return summary


# ============================================================
# 第四部分：命令行接口
# ============================================================

def find_xlsx_files(data_dir: str, pattern: str = '*.xlsx', recursive: bool = True) -> List[str]:
    """递归查找所有 xlsx 文件"""
    if recursive:
        search_pattern = os.path.join(data_dir, '**', pattern)
    else:
        search_pattern = os.path.join(data_dir, pattern)
    return sorted(glob.glob(search_pattern, recursive=recursive))


def load_video_lengths(video_length_path: str) -> Dict[str, float]:
    """加载视频时长信息"""
    if not video_length_path or not os.path.exists(video_length_path):
        return {}
    with open(video_length_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description='CausalHyperGraph Pipeline - 超图构建流水线'
    )
    
    parser.add_argument('--aligned_script_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--video_length', type=str, default=None)
    
    parser.add_argument('--pattern', default='*.xlsx')
    parser.add_argument('--recursive', action='store_true', default=True)
    
    parser.add_argument('--cache_dir', default='ckpt')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_parallel', type=int, default=4)
    
    # 超图构建参数
    parser.add_argument('--time_window', type=int, default=600)
    parser.add_argument('--mo_size', type=int, default=3)
    parser.add_argument('--om_size', type=int, default=3)
    parser.add_argument('--co_window_size', type=int, default=None)
    parser.add_argument('--co_max_size', type=int, default=50)
    parser.add_argument('--similarity_threshold', type=float, default=0.5)
    
    # 严格模式（默认True）
    parser.add_argument('--strict', action='store_true', default=True,
                        help='Enable strict causality (default: True)')
    parser.add_argument('--no_strict', action='store_true',
                        help='Disable strict causality')
    
    parser.add_argument('--min_events', type=int, default=4)
    parser.add_argument('--no_adaptive_window', action='store_true')
    
    # 新增密度控制
    parser.add_argument('--max_mo_ratio', type=float, default=0.3)
    parser.add_argument('--max_om_ratio', type=float, default=0.3)
    parser.add_argument('--enable_debug', action='store_true')
    
    args = parser.parse_args()
    
    strict_causality = False if args.no_strict else args.strict
    
    xlsx_files = find_xlsx_files(args.aligned_script_dir, args.pattern, args.recursive)
    if not xlsx_files:
        print(f"❌ No files found in {args.aligned_script_dir}")
        return
    
    video_list = []
    for f in xlsx_files:
        rel_path = os.path.relpath(f, args.aligned_script_dir)
        video_id = Path(rel_path).stem
        output_subdir = Path(rel_path).parent
        output_path = os.path.join(args.output_dir, str(output_subdir), f"{video_id}.json")
        video_list.append((video_id, f, output_path))
    
    print(f"\n{'='*60}")
    print(f"CausalHyperGraph Pipeline")
    print(f"{'='*60}")
    print(f"  Input: {args.aligned_script_dir}")
    print(f"  Output: {args.output_dir}")
    print(f"  Videos found: {len(video_list)}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max parallel: {args.max_parallel}")
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"  Time window: {args.time_window}s")
    print(f"  MO/OM max size: {args.mo_size}/{args.om_size}")
    print(f"  MO/OM max ratio: {args.max_mo_ratio}/{args.max_om_ratio}")
    print(f"  Strict causality: {strict_causality}")
    print(f"{'='*60}\n")
    
    video_lengths = load_video_lengths(args.video_length) if args.video_length else None
    
    pipeline = HypergraphBuildPipeline(
        cache_dir=args.cache_dir,
        batch_size=args.batch_size
    )
    
    summary = pipeline.run_parallel(
        video_list=video_list,
        max_parallel=args.max_parallel,
        video_lengths=video_lengths,
        time_window=args.time_window,
        mo_size=args.mo_size,
        om_size=args.om_size,
        co_window_size=args.co_window_size,
        co_max_size=args.co_max_size,
        similarity_threshold=args.similarity_threshold,
        strict_causality=strict_causality,
        min_events_for_causal=args.min_events,
        use_adaptive=(not args.no_adaptive_window),
        max_mo_ratio=args.max_mo_ratio,
        max_om_ratio=args.max_om_ratio,
        enable_debug=args.enable_debug,
    )
    
    summary_path = os.path.join(args.output_dir, '_pipeline_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Summary saved to: {summary_path}")


if __name__ == '__main__':
    main()