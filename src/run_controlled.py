"""Run the shared controlled protocol; completed runs are audited before skipping."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import shutil

import pandas as pd

from config import ROOT, SETTINGS, BERT_MODEL
from supcon.joint import ARMS, IMPLEMENTATION, model_fingerprints, run, validate_protocol
from utils.artifacts import environment_info
from utils.controlled_artifacts import read_json, validate_run
from utils.seed import select_torch_device


def summarise(root):
    """A partial progress summary; final delivery requires controlled_report."""
    rows = []
    for file in sorted(root.glob('*/*/seed*/metrics.json')):
        directory = file.parent
        if read_json(directory / 'config.json')['smoke_only']:
            continue
        config, metrics, _ = validate_run(directory)
        rows.append(dict(method=config['method'], setting=config['setting'], seed=config['seed'],
                         macro_f1=metrics['test']['macro_f1'], train_seconds=metrics['train_seconds']))
    if not rows:
        return
    frame = pd.DataFrame(rows)
    frame.to_csv(root / 'runs.csv', index=False)
    frame.groupby(['method', 'setting']).agg(seeds=('seed', 'count'), macro_f1_mean=('macro_f1', 'mean'),
        macro_f1_std=('macro_f1', 'std'), train_seconds_mean=('train_seconds', 'mean')).to_csv(root / 'summary.csv')
    paired = []
    for baseline, treatment in [('ce', 'ce_supcon'), ('ce_supcon', 'ce_supcon_hn')]:
        pairs = frame[frame.method == baseline].merge(frame[frame.method == treatment], on=['setting', 'seed'], suffixes=('_base', '_treat'))
        pairs['delta'] = pairs.macro_f1_treat - pairs.macro_f1_base
        for setting, group in pairs.groupby('setting'):
            paired.append(dict(baseline=baseline, treatment=treatment, setting=setting, seeds=len(group),
                mean_delta=group.delta.mean(), std_delta=group.delta.std() if len(group)>1 else None,
                worst_delta=group.delta.min(), per_seed=[dict(seed=int(r.seed), delta=r.delta) for r in group.itertuples()]))
    (root / 'paired_comparisons.json').write_text(json.dumps(paired, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT/'configs/controlled_v1.json')
    parser.add_argument('--method', nargs='+', choices=ARMS, default=list(ARMS))
    parser.add_argument('--setting', nargs='+', choices=SETTINGS, default=list(SETTINGS))
    parser.add_argument('--seed', type=int, nargs='+', default=[42,43,44])
    parser.add_argument('--model-path', type=Path, default=Path(BERT_MODEL))
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs/controlled_runs')
    parser.add_argument('--device', choices=['auto','cpu','mps','cuda'], default='auto')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--no-save-model', action='store_true')
    parser.add_argument('--retry-incomplete', action='store_true', help='archive incomplete/failed run folders, then restart those runs from step zero')
    parser.add_argument('--smoke', action='store_true', help='one update per arm, 5-shot seed42, small evaluation subsets; never formal results')
    args = parser.parse_args()
    import torch
    if args.threads <= 0:
        parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    device = select_torch_device(args.device)
    protocol = read_json(args.protocol)
    validate_protocol(protocol)
    if args.smoke:
        protocol = {**protocol, 'max_steps':1, 'eval_steps':1}
        args.setting, args.seed = ['5shot'], [42]
        args.output_dir = args.output_dir/'smoke'
    hashes = model_fingerprints(args.model_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    identity = dict(protocol=protocol, model_sha256=hashes, smoke_only=args.smoke,
                    implementation=IMPLEMENTATION, device=device, environment=environment_info())
    frozen = args.output_dir/'protocol.json'
    if frozen.exists() and read_json(frozen) != identity:
        raise ValueError('Output root has a different protocol/model/runtime. Choose a new output directory; do not mix environments.')
    frozen.write_text(json.dumps(identity, indent=2) + '\n', encoding='utf-8')
    for setting in dict.fromkeys(args.setting):
        for seed in dict.fromkeys(args.seed):
            for arm in dict.fromkeys(args.method):
                directory = args.output_dir/arm/setting/f'seed{seed}'
                if directory.exists() and any(directory.iterdir()):
                    status = read_json(directory/'status.json').get('status') if (directory/'status.json').exists() else None
                    if status == 'completed':
                        config, _, _ = validate_run(directory, require_model=not args.no_save_model)
                        if config['protocol'] != protocol or config['model_sha256'] != hashes or config.get('implementation') != IMPLEMENTATION:
                            raise ValueError(f'Completed run identity differs: {directory}')
                        print(f'skip verified completed run {arm}/{setting}/{seed}', flush=True)
                        continue
                    if not args.retry_incomplete:
                        raise ValueError(f'Incomplete run exists: {directory}. Use --retry-incomplete to archive and restart it.')
                    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
                    destination = args.output_dir/'_incomplete'/stamp/arm/setting/directory.name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(directory), str(destination))
                    print(f'archived incomplete run to {destination}', flush=True)
                print(f'run {arm}/{setting}/{seed} on {device}', flush=True)
                try:
                    result = run(arm,setting,seed,protocol=protocol,model_path=args.model_path,model_hashes=hashes,
                                 root=args.output_dir,device=device,save_model=not args.no_save_model,smoke=args.smoke)
                    print(json.dumps(result), flush=True)
                finally:
                    gc.collect()
                    if device == 'cuda':
                        torch.cuda.empty_cache()
    summarise(args.output_dir)


if __name__ == '__main__':
    main()
