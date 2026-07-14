# CausalHyperGraph

A **type-aware hypergraph neural network** framework for causal reasoning in video stories. This framework constructs causal hypergraphs with three semantic hyperedge types (MO/OM/CO), combined with hyperedge-aware retrieval and hierarchical readout mechanisms to enable multi-cause and multi-effect reasoning in complex narrative videos.

---

## Key Features

- **Three Semantic Hyperedge Types**
  - **MO** (Multiple-causes-to-One-effect): Hyperedges connecting multiple cause events to a single effect
  - **OM** (One-cause-to-Multiple-effects): Hyperedges connecting a single cause to multiple effects
  - **CO** (Character Co-occurrence): Hyperedges capturing character interaction patterns

- **Type-Aware Hypergraph Convolution**
  - Independent transformation matrices Θ_τ for each hyperedge type
  - Adaptive type attention aggregation conditioned on question embeddings

- **Hyperedge-Aware Reasoner (HAR)**
  - Seed node retrieval → Hyperedge expansion → Hierarchical readout
  - Dual attention at both hyperedge-level and node-level

- **Two-Stage Training Strategy**
  - **Pretraining**: Hyperedge prediction with type-aware contrastive learning
  - **Finetuning**: End-to-end QA optimization with joint loss L = L_fine + λ₁L_pre + λ₂L_aux

- **Multi-stage Heuristic Causality Verification (MHCV)**
  - Temporal ordering → Character continuity → Causal keyword matching → Embedding semantic similarity

---

## Project Structure

```
CausalHyperGraph/
├── causal_hypergraph/
│   └── models.py                 # Core model definitions
│       ├── TypeAwareHypergraphConv   # Type-aware hypergraph convolution layer
│       ├── HGNE                      # Hypergraph neural encoder
│       └── EndToEndModel             # End-to-end QA model
├── hcm/                          # Hypergraph Construction Module
│   ├── build_hypergraph.py       # Core hypergraph builder with MHCV
│   └── run_pipeline_build_hypergraph.py  # Data processing pipeline
├── hgne/                         # Hypergraph Neural Encoder
│   ├── train_hgne.py             # Two-stage pretraining/finetuning
│   └── train_with_qa_head.py     # Training with QA classification head
├── har/                          # Hyperedge-Aware Reasoner
│   ├── hypergraph_readout.py     # Hypergraph readout (attention/hierarchical)
│   ├── question_type_infer.py    # Question type inference & auxiliary loss
│   ├── run_qa.py                 # Unified inference script (classifier/LLM dual mode)
│   ├── ablation_runner.py        # Automated ablation experiments
│   ├── build_special_testset.py  # MTO/OMT specialized testset construction
│   └── visualize_hypergraph.py   # Hyperedge visualization tools
├── train_e2e.py                  # End-to-end finetuning (full loss implementation)
└── README.md                     # This file
```

---

## Installation

```bash
pip install torch torchvision torchaudio
pip install transformers tqdm numpy pandas
pip install openpyxl  # Excel parsing
pip install matplotlib networkx scipy  # Visualization
```

> Uses `bert-large-uncased` as the default text encoder. Set `HF_ENDPOINT` to configure mirror sources if needed.

---

## Usage

### 1. Hypergraph Construction (HCM)

Build causal hypergraphs from aligned script/subtitle Excel files:

```bash
python hcm/run_pipeline_build_hypergraph.py \
    --aligned_script_dir data/aligned_script/GOT \
    --output_dir experiments/hypergraphs \
    --batch_size 32 \
    --max_parallel 6 \
    --strict \
    --mo_size 3 \
    --om_size 3 \
    --max_mo_ratio 0.3 \
    --max_om_ratio 0.3
```

**Key Parameters:**

| Parameter | Description | Default |
|:---|:---|:---|
| `--strict` / `--no_strict` | Strict / relaxed causality verification | `True` |
| `--mo_size` | Max cause nodes per MO hyperedge | 3 |
| `--om_size` | Max effect nodes per OM hyperedge | 3 |
| `--max_mo_ratio` | Max MO edges as ratio of total events | 0.3 |
| `--max_om_ratio` | Max OM edges as ratio of total events | 0.3 |
| `--time_window` | Causal time window in seconds | 600 |
| `--no_adaptive_window` | Disable adaptive time window | - |

Each video produces a JSON file containing nodes (events) and typed hyperedges.

---

### 2. HGNE Pretraining

Two-stage training on constructed hypergraphs:

```bash
python hgne/train_hgne.py \
    --hypergraph_dir experiments/hypergraphs \
    --output_dir checkpoints/hgne \
    --pretrain_epochs 20 \
    --finetune_epochs 50 \
    --hidden_dim 512 \
    --num_layers 3 \
    --lr_pretrain 2e-5 \
    --lr_finetune 1e-4 \
    --lambda_pre 0.3 \
    --lambda_aux 0.1
```

**Training Phases:**
1. **Pretraining** (20 epochs): `HyperedgePredictionLoss` for hyperedge existence prediction
2. **Finetuning** (50 epochs): Joint `TypeAwareContrastiveLoss` + `HyperedgePredictionLoss` + `TypePredictionLoss`

Output: `checkpoints/hgne/hgne_finetuned.pt`

---

### 3. End-to-End Finetuning (E2E)

Load pretrained HGNE and attach QA head for end-to-end finetuning:

```bash
python train_e2e.py \
    --questions data/qa_questions.json \
    --hypergraph_dir experiments/hypergraphs \
    --hgne_checkpoint checkpoints/hgne/hgne_finetuned.pt \
    --output_dir checkpoints/e2e_final \
    --epochs 30 \
    --lr 5e-5 \
    --lambda_pre 0.3 \
    --lambda_aux 0.1 \
    --gradient_accumulation 4
```

**Loss Function:**
```
L = L_fine (CrossEntropy) + λ₁ * L_pre (hyperedge prediction) + λ₂ * L_aux (question type auxiliary)
```

---

### 4. Inference (QA)

Supports direct classifier inference or LLM-enhanced reasoning (Qwen2.5-3B):

```bash
# Classifier-only mode
python har/run_qa.py \
    --questions data/test_questions.json \
    --hypergraph_dir experiments/hypergraphs \
    --checkpoint checkpoints/e2e_final/best_model.pt \
    --output results/e2e_results.json \
    --model_type e2e \
    --no_llm

# LLM-enhanced mode
python har/run_qa.py \
    --questions data/test_questions.json \
    --hypergraph_dir experiments/hypergraphs \
    --checkpoint checkpoints/hgne/hgne_finetuned.pt \
    --output results/har_llm_results.json \
    --model_type hgne \
    --use_local_llm \
    --top_k 10 \
    --M 5
```

---

### 5. Ablation Studies

Automated ablation experiments (Section 4.4):

```bash
python har/ablation_runner.py \
    --questions data/test_questions.json \
    --hypergraph_dir experiments/hypergraphs \
    --checkpoint checkpoints/e2e_final/best_model.pt \
    --output_dir experiments/ablation_results
```

Supported configurations:
- BERT baseline (no graph structure)
- Plain graph (PlotTree structure)
- Hypergraph without type awareness
- Type-aware HGNE
- + Hyperedge-aware retrieval HAR
- + Pretraining + finetuning (full model)

---

### 6. Specialized Testset Construction

Build MTO-Test (multi-cause-to-one-effect) and OMT-Test (one-cause-to-multi-effect) benchmarks:

```bash
python har/build_special_testset.py \
    --questions data/all_questions.json \
    --hypergraph_dir experiments/hypergraphs \
    --output_dir experiments/special_testsets \
    --max_samples 2400
```

---

### 7. Hypergraph Visualization

Visualize individual hyperedges or hypergraph summaries:

```bash
# Summary visualization for all types
python har/visualize_hypergraph.py \
    --hypergraph experiments/hypergraphs/xxx.json \
    --output_dir experiments/visualizations \
    --edge_type all

# Specific MO hyperedge
python har/visualize_hypergraph.py \
    --hypergraph experiments/hypergraphs/xxx.json \
    --output_dir experiments/visualizations \
    --edge_type MO \
    --edge_index 0
```

---

## Data Format

### Event Node (Excel → JSON)

```json
{
  "id": "e_0001",
  "time_seconds": 120.5,
  "start_time": "00:02:00",
  "end_time": "00:02:30",
  "P_i": ["Character A", "Character B"],
  "A_i": "Action description",
  "L_i": "Scene location",
  "S_i": "Event summary",
  "text": "Full text",
  "embedding": [1024-dim vector]
}
```

### Hyperedge

```json
{
  "type": "MO",
  "nodes": ["e_0001", "e_0002", "e_0003"],
  "source_nodes": ["e_0001", "e_0002"],
  "target_nodes": ["e_0003"],
  "weight": 0.85,
  "causality_score": 0.82
}
```

### Question

```json
{
  "vid": "video_id",
  "question": "What caused the character to leave?",
  "choices": ["Option A", "Option B", "Option C", "Option D"],
  "option": "A",
  "type": "mto"
}
```

---

## Core Modules

### TypeAwareHypergraphConv

Implements the type-aware hypergraph convolution (Equation 4):

```
X^(l+1) = Σ_τ α_τ · D_v^(-1/2) · H_τ · W_e · D_e^(-1) · H_τ^T · D_v^(-1/2) · X^(l) · Θ_τ
```

Attention weights α_τ are computed via interaction between question embeddings and type embeddings.

### HyperedgeAwareReasoner

Reasoning pipeline:
1. **Seed Retrieval**: Select top-k seed nodes by question-node cosine similarity
2. **Hyperedge Expansion**: Retrieve relevant hyperedges connected to seeds (up to M)
3. **Hierarchical Readout**:
   - Hyperedge-level: Multi-head attention aggregation over hyperedge embeddings
   - Node-level: Attention readout within the most important hyperedge

### CausalityVerifier (MHCV)

Four-stage heuristic verification:

| Stage | Check | Weight |
|:---|:---|:---|
| 1 | Temporal ordering (cause must precede effect) | Must pass |
| 2 | Character continuity | +0.15 |
| 3 | Causal keyword matching (strong/weak/emotional) | +0.30 |
| 4 | Embedding semantic similarity | +0.35 |

---

## Performance Optimizations

- **GPU**: Automatic TF32, CuDNN benchmark, mixed-precision training (AMP)
- **Pipeline Parallelism**: CPU preprocessing parallelized with GPU encoding
- **Density Control**: `max_mo_ratio` / `max_om_ratio` prevent overly dense graphs
- **Adaptive Time Window**: Automatically adjusts causal window by video genre (drama/suspense/comedy/epic)

---

## Citation

```bibtex
@article{causalhypergraph2025,
  title={CausalHyperGraph: Type-Aware Hypergraph Neural Networks for Video Story Causal Reasoning},
  author={Your Name},
  journal={arXiv preprint},
  year={2025}
}
```
