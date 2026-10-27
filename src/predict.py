"""Offline inference from a selected controlled-run checkpoint. Scores are uncalibrated."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import pandas as pd
import torch

from supcon.joint import load_saved_model, synchronize
from utils.controlled_artifacts import file_hashes, read_json
from utils.precision import resolve_precision
from utils.seed import select_torch_device


class IntentPredictor:
    def __init__(self, run_dir, *, device='auto'):
        from transformers import AutoTokenizer
        self.run_dir = Path(run_dir)
        config = read_json(self.run_dir / 'config.json')
        if read_json(self.run_dir / 'status.json').get('status') != 'completed':
            raise ValueError('Only completed checkpoints can be used')
        if config.get('smoke_only'):
            raise ValueError('Smoke checkpoints are not inference deliverables')
        model_dir = self.run_dir / 'model'
        if read_json(model_dir / 'manifest.json') != file_hashes(model_dir):
            raise ValueError('Saved model checksum mismatch')
        self.device = select_torch_device(device)
        self.precision = resolve_precision(config['protocol']['precision'], self.device)['resolved']
        self.model, self.categories = load_saved_model(self.run_dir, self.device)
        if self.categories != config['categories']:
            raise ValueError('Saved label order differs from training config')
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.max_length = config['protocol']['max_length']
        self.method = config['method']

    def predict(self, texts, *, top_k=3, batch_size=32):
        if not texts or any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError('Every input must be a nonempty text string')
        if not 1 <= top_k <= len(self.categories) or batch_size < 1:
            raise ValueError('Invalid top_k or batch_size')
        rows = []
        self.model.eval()
        synchronize(self.device)
        start = time.perf_counter()
        with torch.inference_mode():
            for offset in range(0, len(texts), batch_size):
                batch = texts[offset:offset+batch_size]
                lengths = [len(ids) for ids in self.tokenizer(batch, truncation=False, verbose=False)['input_ids']]
                encoded = self.tokenizer(batch, padding=True, truncation=True, max_length=self.max_length, return_tensors='pt')
                with torch.autocast(device_type=self.device, dtype=torch.bfloat16, enabled=self.precision=='bf16'):
                    pooled = self.model.embed(encoded['input_ids'].to(self.device), encoded['attention_mask'].to(self.device))
                    logits = self.model.classify(pooled)
                scores = torch.softmax(logits.float(), dim=1)
                if not torch.isfinite(scores).all():
                    raise FloatingPointError('Nonfinite prediction scores')
                # Keep the same tie policy as training selection and CSV export.
                indices = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :top_k]
                values = scores.gather(1, indices)
                for text, length, score_row, ids in zip(batch, lengths, values.cpu().tolist(), indices.cpu().tolist()):
                    rows.append(dict(text=text, label=self.categories[ids[0]], truncated=length>self.max_length,
                        top_k=[dict(label=self.categories[i], score=s) for i,s in zip(ids,score_row)]))
        synchronize(self.device)
        elapsed = time.perf_counter()-start
        return dict(method=self.method, device=self.device, precision=self.precision, max_length=self.max_length,
                    score_note='uncalibrated softmax scores; not a reliability guarantee',
                    elapsed_seconds=elapsed, examples_per_second=len(texts)/elapsed,
                    timing='includes tokenization and inference; excludes model loading; first call may include warmup',
                    predictions=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    checkpoint = parser.add_mutually_exclusive_group(required=True)
    checkpoint.add_argument('--run-dir', type=Path)
    checkpoint.add_argument('--selection', type=Path, help='selected_model.json produced by controlled_report')
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--text', action='append', help='repeat for multiple inputs')
    inputs.add_argument('--input', type=Path, help='UTF-8 CSV with a text column')
    parser.add_argument('--output', type=Path, help='write one JSON document')
    parser.add_argument('--device', choices=['auto','cpu','mps','cuda'], default='auto')
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    directory = args.run_dir
    if args.selection:
        directory = (args.selection.parent / read_json(args.selection)['run_dir']).resolve()
    texts = args.text if args.text else pd.read_csv(args.input, keep_default_na=False).text.tolist()
    predictor = IntentPredictor(directory, device=args.device)
    payload = predictor.predict(texts, top_k=args.top_k, batch_size=args.batch_size)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding='utf-8')
    else:
        print(rendered, end='')


if __name__ == '__main__':
    main()
