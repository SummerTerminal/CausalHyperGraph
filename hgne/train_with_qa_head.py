"""
train_with_qa_head.py - two-stage training with QA head, using unified training functions
"""
import os, json, argparse, torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
import random
# 统一训练函数和数据集
from train_e2e import train_with_full_loss, TypeAwareQADataset, collate_fn, smart_load_json
from models import HGNE, EndToEndModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--questions', required=True)
    parser.add_argument('--hypergraph_dir', required=True)
    parser.add_argument('--hgne_checkpoint', required=True)
    parser.add_argument('--output_dir', default='checkpoints/e2e_full')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gradient_accumulation', type=int, default=4)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--lambda_pre', type=float, default=0.3)
    parser.add_argument('--lambda_aux', type=float, default=0.1)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--cache_dir', default='ckpt')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*60}\nEnd-to-end HGNE training (unified)\n{'='*60}")
    print(f"Device: {dev}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    questions = smart_load_json(args.questions)
    print(f"\nLoaded {len(questions)} questions")

    tok = AutoTokenizer.from_pretrained('bert-large-uncased', cache_dir=args.cache_dir)
    bert = AutoModel.from_pretrained('bert-large-uncased', cache_dir=args.cache_dir)
    bert.to(dev).eval()

    print(f"\nLoading HGNE: {args.hgne_checkpoint}")
    hgne = HGNE(in_channels=1024, hidden_channels=args.hidden_dim, num_layers=args.num_layers)
    ckpt = torch.load(args.hgne_checkpoint, map_location=dev)
    sd = ckpt.get('model_state_dict', ckpt)
    if all(k.startswith('module.') for k in sd):
        sd = {k[7:]: v for k, v in sd.items()}
    if 'q_proj.weight' not in sd:
        print("  q_proj not in checkpoint, init with identity")
        with torch.no_grad():
            d = min(1024, args.hidden_dim)
            hgne.q_proj.weight.data[:, :d] = torch.eye(d)
            hgne.q_proj.bias.data.zero_()
    hgne.load_state_dict(sd, strict=False)
    hgne.to(dev)

    model = EndToEndModel(hgne, hidden_dim=args.hidden_dim, num_answers=5,
                          use_har=True, top_k_seed=10, M=5)
    model.to(dev)
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")

    ds = TypeAwareQADataset(questions, args.hypergraph_dir, tok, bert, dev)
    train_sz = int(0.8 * len(ds))
    val_sz = len(ds) - train_sz
    gen = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_sz, val_sz], generator=gen)
    print(f"Train: {train_sz}, Val: {val_sz}")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          collate_fn=collate_fn, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)

    model, best_acc, hist = train_with_full_loss(
        model, train_dl, val_dl, args.epochs, dev, args.output_dir,
        lr=args.lr, lambda_pre=args.lambda_pre, lambda_aux=args.lambda_aux,
        patience=args.patience, grad_accum=args.gradient_accumulation
    )

    with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
        json.dump(hist, f, indent=2)

    print(f"\n{'='*60}\nTraining done! Best val acc: {best_acc:.2f}%\nResults: {args.output_dir}\n{'='*60}")

if __name__ == '__main__':
    main()