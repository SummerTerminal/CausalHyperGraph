# run_pipeline_build_hypergraph.py - data processing & parallel scheduling
"""
Pipeline: read aligned_script Excel, extract events, BERT embeddings, build hypergraph.
Supports GOT/Movie and BigBang/Friends formats.
"""
import os, json, argparse, time, pandas as pd, torch, numpy as np, sys, io
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import glob, re
from typing import List, Dict, Tuple, Optional

from build_hypergraph import build_hypergraph

# fix windows git bash encoding
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except: pass

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# ============================================================
# Part 1: Excel parsing helpers
# ============================================================

def parse_time_to_seconds(time_str: str) -> Optional[float]:
    """Convert timestamp string to seconds"""
    if not time_str or not isinstance(time_str, str): return None
    time_str = time_str.strip()
    if not time_str: return None
    time_str = time_str.replace(',', '.')
    m = re.match(r'(\d+):(\d+):(\d+)(?:\.(\d+))?', time_str)
    if m:
        h, mi, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
        ms = m.group(4)
        if ms:
            ms = ms.ljust(3, '0')[:3]
            s += int(ms) / 1000.0
        return h * 3600 + mi * 60 + s
    return None

def is_valid_string(s: str) -> bool:
    """check if string is meaningful"""
    if not s or not isinstance(s, str): return False
    s = s.strip()
    if not s: return False
    # 常见的无效占位符
    if s.lower() in {'nan','none','null','na','-','--','n/a',''}: return False
    return True

def detect_sheet_format(xl: pd.ExcelFile) -> Dict[str, str]:
    """detect which sheets contain subtitle and script data"""
    sheets = xl.sheet_names
    sub_sheet, scr_sheet = None, None
    for s in sheets:
        if s.startswith('subtitle_'): sub_sheet = s
        elif s.startswith('script_') and not s.startswith('scripts'): scr_sheet = s
    if sub_sheet is None and 'subtitle' in sheets: sub_sheet = 'subtitle'
    if scr_sheet is None:
        if 'scripts' in sheets: scr_sheet = 'scripts'
        elif 'script' in sheets: scr_sheet = 'script'
    return {'subtitle_sheet': sub_sheet, 'script_sheet': scr_sheet}

def generate_summary(content: str, action: str, characters: List[str]) -> str:
    """
    Generate a short summary for an event.
    Uses LLM if env var is set, otherwise simple heuristic.
    """
    use_llm = os.getenv('USE_LLM_SUMMARY', 'false').lower() == 'true'
    if use_llm:
        try:
            import google.generativeai as genai
            api_key = os.getenv('GEMINI_API_KEY')
            if api_key:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-2.0-flash')
                prompt = f"Summarize this movie scene in one sentence (max 50 words): {content[:1000]}"
                resp = model.generate_content(prompt)
                return resp.text.strip()
        except Exception as e:
            print(f"  LLM summary failed: {e}, fallback to heuristic")

    # 规则摘要：取首句 + 角色
    first_sent = content.split('.')[0] if '.' in content else content
    summary = first_sent[:250]
    if characters: summary = f"{' '.join(characters[:3])}: {summary}"
    return summary

def extract_scenes_from_excel(xlsx_path: str) -> List[Dict]:
    """
    Extract scene/event list from Excel.
    Key fix: use results sheet index_result column to map subtitle timestamps to script rows.
    """
    xl = pd.ExcelFile(xlsx_path)
    sheets_info = detect_sheet_format(xl)
    if not sheets_info['script_sheet']:
        print(f"  ⚠️ No script sheet in {xlsx_path}")
        return []

    script_df = pd.read_excel(xl, sheet_name=sheets_info['script_sheet'])

    # 寻找对齐结果表
    results_sheet = None
    for s in xl.sheet_names:
        if s.startswith('results_') and 'human' not in s:
            results_sheet = s; break
    if not results_sheet:
        for s in xl.sheet_names:
            if s == 'results': results_sheet = s; break
    if not results_sheet:
        for s in xl.sheet_names:
            if 'results_human' in s:
                results_sheet = s
                print(f"  ⚠️ Using human-annotated alignment: {results_sheet}")
                break

    time_map = {}
    if results_sheet:
        try:
            res_df = pd.read_excel(xl, sheet_name=results_sheet)
            for _, row in res_df.iterrows():
                idx = row.get('index_result')
                if idx is None or pd.isna(idx): continue
                idx = int(idx)
                if idx not in time_map:
                    time_map[idx] = {
                        'start_time': str(row.get('start_time','')).strip(),
                        'end_time': str(row.get('end_time','')).strip(),
                        'dialog': str(row.get('dialog','')).strip()
                    }
                else:
                    time_map[idx]['end_time'] = str(row.get('end_time','')).strip()
        except Exception as e:
            print(f"  ⚠️ Could not read results sheet '{results_sheet}': {e}")

    events = []
    cur_scene = 1
    cur_loc = ""
    cur_desc = ""

    def _clean(s):
        s = str(s).strip()
        return '' if s.lower() in ['nan','none',''] else s

    for idx, row in script_df.iterrows():
        # excel row number for alignment
        if 'Unnamed: 0' in script_df.columns:
            excel_row_num = int(row['Unnamed: 0'])
        else:
            excel_row_num = idx + 1

        rec_type = str(row.get('record_type','')).strip().lower()
        chars = _clean(str(row.get('characters','')))
        loc = _clean(str(row.get('location','')))
        content = _clean(str(row.get('content','')))
        desc = _clean(str(row.get('description','')))

        if rec_type == 'scene':
            if 'scene_index' in script_df.columns:
                si = row.get('scene_index')
                if si and not pd.isna(si): cur_scene = int(si)
            if loc: cur_loc = loc
            if desc: cur_desc = desc
            continue

        if rec_type in ['dialog',''] or not rec_type:
            if not content:
                if desc: content = desc
                else: continue

            # 角色列表
            if chars:
                char_list = [c.strip() for c in re.split(r'[,/&]', chars) if is_valid_string(c.strip())]
            else:
                char_list = []

            ts = time_map.get(excel_row_num, {})
            start_time = ts.get('start_time','')
            end_time = ts.get('end_time','')
            t_sec = parse_time_to_seconds(start_time)

            # build full text
            parts = []
            if cur_loc: parts.append(f"Location: {cur_loc}.")
            if char_list: parts.append(f"Characters: {', '.join(char_list)}.")
            if content: parts.append(f"Dialog: {content}")
            if desc: parts.append(f"Description: {desc}")
            full_text = ' '.join(parts)

            summary = generate_summary(content, content[:200], char_list)

            events.append({
                'id': '',
                'time_seconds': t_sec,
                'start_time': start_time,
                'end_time': end_time,
                'P_i': char_list,
                'A_i': content[:200] if len(content) > 200 else content,
                'L_i': cur_loc,
                'S_i': summary,
                'text': full_text,
                'description': desc,
                'scene_index': cur_scene,
            })

    # assign ids
    for i, e in enumerate(events):
        e['id'] = f'e_{i:04d}'

    t_cnt = sum(1 for e in events if e['time_seconds'] is not None)
    print(f"  📊 Extracted {len(events)} events, {t_cnt}/{len(events)} ({100*t_cnt/max(1,len(events)):.1f}%) with timestamps")
    return events

# ============================================================
# Part 2: BERT Embedding Encoder (GPU)
# ============================================================

class EmbeddingEncoder:
    """BERT-large-uncased encoder, batch inference on GPU"""
    def __init__(self, cache_dir='ckpt', batch_size=32):
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading bert-large-uncased on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased', cache_dir=cache_dir)
        self.model = AutoModel.from_pretrained('bert-large-uncased', cache_dir=cache_dir)
        self.model.to(self.device).eval()
        print(f"Model loaded on {self.device}")

    @torch.no_grad()
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        inp = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inp = {k: v.to(self.device) for k, v in inp.items()}
        out = self.model(**inp)
        return out.last_hidden_state[:, 0, :].cpu().numpy()

    def encode_events(self, events: List[Dict]) -> List[Dict]:
        texts = []
        for e in events:
            t = e.get('text', '')
            if not t or not t.strip(): t = e.get('S_i', 'No text')
            texts.append(t)

        all_emb = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            batch_emb = self.encode_batch(batch)
            all_emb.append(batch_emb)
        if all_emb:
            all_emb = np.concatenate(all_emb, axis=0)
        else:
            all_emb = np.array([])

        for i, e in enumerate(events):
            e['embedding'] = all_emb[i].tolist() if i < len(all_emb) else []
        return events

# ============================================================
# Part 3: Pipeline scheduler
# ============================================================

class HypergraphBuildPipeline:
    """Pipeline: CPU extraction, GPU embedding, hypergraph construction"""
    def __init__(self, cache_dir='ckpt', batch_size=32):
        self.encoder = EmbeddingEncoder(cache_dir=cache_dir, batch_size=batch_size)

    def process_single_video(self, xlsx_path, output_path, video_id,
                             video_lengths=None, **kwargs):
        """Process one video, save hypergraph JSON"""
        if os.path.exists(output_path):
            print(f"  ⏭️ {video_id}: already exists, skipping")
            return None
        t0 = time.time()
        events = extract_scenes_from_excel(xlsx_path)
        if not events:
            print(f"  ⚠️ {video_id}: no events extracted")
            return None
        events = self.encoder.encode_events(events)
        MO, OM, CO, stats = build_hypergraph(events=events, video_id=video_id,
                                            video_lengths=video_lengths, **kwargs)
        hg = {
            'video_id': video_id,
            'nodes': [{
                'id': e['id'], 'time_seconds': e.get('time_seconds'),
                'start_time': e.get('start_time',''), 'end_time': e.get('end_time',''),
                'P_i': e.get('P_i',[]), 'A_i': e.get('A_i',''), 'L_i': e.get('L_i',''),
                'S_i': e.get('S_i',''), 'text': e.get('text',''), 'embedding': e.get('embedding',[])
            } for e in events],
            'hyperedges': {'MO': MO, 'OM': OM, 'CO': CO},
            'stats': stats
        }
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(hg, f, ensure_ascii=False, indent=2)
        elapsed = time.time() - t0
        print(f"  ✅ {video_id}: {stats['num_events']} events, MO:{stats['num_mo_edges']} OM:{stats['num_om_edges']} CO:{stats['num_co_edges']} ({elapsed:.1f}s)")
        return stats

    def run_parallel(self, video_list, max_parallel=4, video_lengths=None, **kwargs):
        """Parallel processing: phase1 preprocessing, phase2 encoding+build"""
        total = len(video_list)
        results, errors, skipped = [], [], 0
        import concurrent.futures

        def pre_task(vid, xlsx, out):
            if os.path.exists(out): return ('skip', vid, None, None)
            try:
                evts = extract_scenes_from_excel(xlsx)
                return ('ready', vid, evts, out)
            except Exception as e:
                return ('error', vid, str(e), None)

        print(f"\n{'='*60}\nPhase 1: Preprocessing {total} videos (max_parallel={max_parallel})\n{'='*60}")
        pre_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as ex:
            futures = {ex.submit(pre_task, v, x, o): v for v, x, o in video_list}
            for fut in concurrent.futures.as_completed(futures):
                status, vid, data, out = fut.result()
                if status == 'skip': skipped += 1
                elif status == 'error':
                    errors.append(vid)
                    print(f"  ❌ {vid}: {data}")
                else:
                    pre_results.append((vid, data, out))

        print(f"\n{'='*60}\nPhase 2: Encoding + Building ({len(pre_results)} videos)\n{'='*60}")
        for i, (vid, evts, out) in enumerate(pre_results):
            print(f"\n[{i+1}/{len(pre_results)}] Processing {vid}...")
            try:
                evts = self.encoder.encode_events(evts)
                MO, OM, CO, stats = build_hypergraph(events=evts, video_id=vid,
                                                    video_lengths=video_lengths, **kwargs)
                hg = {
                    'video_id': vid,
                    'nodes': [{
                        'id': e['id'], 'time_seconds': e.get('time_seconds'),
                        'start_time': e.get('start_time',''), 'end_time': e.get('end_time',''),
                        'P_i': e.get('P_i',[]), 'A_i': e.get('A_i',''), 'L_i': e.get('L_i',''),
                        'S_i': e.get('S_i',''), 'text': e.get('text',''), 'embedding': e.get('embedding',[])
                    } for e in evts],
                    'hyperedges': {'MO': MO, 'OM': OM, 'CO': CO},
                    'stats': stats
                }
                Path(out).parent.mkdir(parents=True, exist_ok=True)
                with open(out, 'w', encoding='utf-8') as f:
                    json.dump(hg, f, ensure_ascii=False, indent=2)
                results.append(stats)
                print(f"  ✅ {vid}: {stats['num_events']} events, MO:{stats['num_mo_edges']} OM:{stats['num_om_edges']} CO:{stats['num_co_edges']}")
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
        print(f"\n{'='*60}\nPipeline Complete\n{'='*60}")
        print(f"  Total: {total}, Processed: {len(results)}, Skipped: {skipped}, Errors: {len(errors)}")
        print(f"  Events: {summary['total_events']}, MO: {summary['total_mo_edges']}, OM: {summary['total_om_edges']}, CO: {summary['total_co_edges']}")
        print(f"{'='*60}\n")
        return summary

# ============================================================
# Part 4: CLI
# ============================================================

def find_xlsx_files(data_dir, pattern='*.xlsx', recursive=True):
    """recursively find all xlsx files under data_dir"""
    search = os.path.join(data_dir, '**', pattern) if recursive else os.path.join(data_dir, pattern)
    return sorted(glob.glob(search, recursive=recursive))

def load_video_lengths(path):
    if not path or not os.path.exists(path): return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description='CausalHyperGraph Pipeline')
    parser.add_argument('--aligned_script_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--video_length', type=str, default=None)
    parser.add_argument('--pattern', default='*.xlsx')
    parser.add_argument('--recursive', action='store_true', default=True)
    parser.add_argument('--cache_dir', default='ckpt')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_parallel', type=int, default=4)
    parser.add_argument('--time_window', type=int, default=600)
    parser.add_argument('--mo_size', type=int, default=3)
    parser.add_argument('--om_size', type=int, default=3)
    parser.add_argument('--co_window_size', type=int, default=None)
    parser.add_argument('--co_max_size', type=int, default=50)
    parser.add_argument('--similarity_threshold', type=float, default=0.5)
    parser.add_argument('--strict', action='store_true', default=True)
    parser.add_argument('--no_strict', action='store_true')
    parser.add_argument('--min_events', type=int, default=4)
    parser.add_argument('--no_adaptive_window', action='store_true')
    parser.add_argument('--max_mo_ratio', type=float, default=0.3)
    parser.add_argument('--max_om_ratio', type=float, default=0.3)
    parser.add_argument('--enable_debug', action='store_true')
    args = parser.parse_args()

    strict = False if args.no_strict else args.strict

    xlsx_files = find_xlsx_files(args.aligned_script_dir, args.pattern, args.recursive)
    if not xlsx_files:
        print(f"❌ No files found in {args.aligned_script_dir}")
        return

    video_list = []
    for f in xlsx_files:
        rel = os.path.relpath(f, args.aligned_script_dir)
        vid = Path(rel).stem
        out_sub = Path(rel).parent
        out = os.path.join(args.output_dir, str(out_sub), f"{vid}.json")
        video_list.append((vid, f, out))

    print(f"\n{'='*60}\nCausalHyperGraph Pipeline\n{'='*60}")
    print(f"  Input: {args.aligned_script_dir}")
    print(f"  Output: {args.output_dir}")
    print(f"  Videos found: {len(video_list)}")
    print(f"  Batch size: {args.batch_size}, Max parallel: {args.max_parallel}")
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"  Time window: {args.time_window}s, MO/OM size: {args.mo_size}/{args.om_size}, ratios: {args.max_mo_ratio}/{args.max_om_ratio}")
    print(f"  Strict causality: {strict}")
    print(f"{'='*60}\n")

    vlengths = load_video_lengths(args.video_length) if args.video_length else None
    pipeline = HypergraphBuildPipeline(cache_dir=args.cache_dir, batch_size=args.batch_size)
    summary = pipeline.run_parallel(
        video_list=video_list,
        max_parallel=args.max_parallel,
        video_lengths=vlengths,
        time_window=args.time_window,
        mo_size=args.mo_size, om_size=args.om_size,
        co_window_size=args.co_window_size, co_max_size=args.co_max_size,
        similarity_threshold=args.similarity_threshold,
        strict_causality=strict,
        min_events_for_causal=args.min_events,
        use_adaptive=(not args.no_adaptive_window),
        max_mo_ratio=args.max_mo_ratio, max_om_ratio=args.max_om_ratio,
        enable_debug=args.enable_debug,
    )
    sum_path = os.path.join(args.output_dir, '_pipeline_summary.json')
    with open(sum_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Summary saved to: {sum_path}")

if __name__ == '__main__':
    main()