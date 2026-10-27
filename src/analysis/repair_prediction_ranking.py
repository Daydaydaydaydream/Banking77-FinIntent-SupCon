"""Recover existing controlled exports after the tied-score ranking bug; never retrain."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from analysis.audit_artifacts import frame_digest
from config import SETTINGS
from supcon.joint import ARMS, load_saved_model, predict_dataset, synchronize
from supcon.model import TextDataset
from utils.artifacts import RunArtifacts, environment_info
from utils.controlled_artifacts import file_hashes, read_json, validate_run
from utils.data import encode_labels, load_splits
from utils.metrics import classification_metrics
from utils.seed import select_torch_device

RANKING = 'score-descending-class-id-ascending-v1'
REPLACED = ('metrics.json', 'predictions.csv', 'validation_predictions.csv', 'confusion_matrix.csv')


def repair(root, *, settings=('5shot',), seeds=(42,43,44), device='cuda'):
    root = Path(root).resolve()
    device = select_torch_device(device)
    identity = read_json(root/'protocol.json')
    if identity['smoke_only']:
        raise ValueError('This recovery command is for formal runs only')
    if identity.get('device') != device or identity.get('environment') != environment_info():
        raise ValueError('Recovery must use the original training device/runtime; do not change the environment')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    records = []
    # Verify ALL replacements before touching any original. If a GPU rerun differs beyond
    # exact score ties or cannot reproduce the selected validation F1, stop without repair.
    with tempfile.TemporaryDirectory(prefix='.ranking-repair-', dir=root) as tmp:
        staged = Path(tmp)
        for setting in dict.fromkeys(settings):
            for seed in sorted(set(seeds)):
                for arm in ARMS:
                    directory = root/arm/setting/f'seed{seed}'
                    config = read_json(directory/'config.json')
                    original = read_json(directory/'metrics.json')
                    if config['smoke_only']:
                        raise ValueError(f'Not a formal run: {directory}')
                    for key in ('protocol', 'model_sha256', 'implementation'):
                        if config.get(key) != identity.get(key):
                            raise ValueError(f'Run identity mismatch: {directory}')
                    if read_json(directory/'env.json') != identity['environment']:
                        raise ValueError(f'Run environment mismatch: {directory}')
                    try:
                        validate_run(directory)
                    except ValueError as exc:
                        if not str(exc).startswith('Selected validation score mismatch:'):
                            raise
                    if original.get('prediction_ranking') == RANKING:
                        validate_run(directory, require_model=True)
                        print(f'already verified {arm}/{setting}/{seed}', flush=True)
                        continue
                    print(f're-export saved model {arm}/{setting}/{seed}', flush=True)
                    model_dir = directory/'model'
                    if read_json(model_dir/'manifest.json') != file_hashes(model_dir):
                        raise ValueError(f'Saved model checksum mismatch: {directory}')
                    model, categories = load_saved_model(directory, device)
                    if categories != config['categories']:
                        raise ValueError(f'Category order mismatch: {directory}')
                    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
                    train, val, test, source_categories = load_splits(setting,seed)
                    if source_categories != categories or config['data_fingerprints'] != {
                        k: frame_digest(v) for k,v in [('train',train),('validation',val),('test',test)]}:
                        raise ValueError(f'Data fingerprint mismatch: {directory}')
                    target = RunArtifacts(arm,setting,seed,root=staged)
                    for name in ('config.json','env.json','status.json','training_history.json'):
                        shutil.copy2(directory/name,target.path/name)
                    metrics, changes = {}, {}
                    synchronize(device)
                    started = time.perf_counter()
                    for split, frame, filename in [('validation',val,'validation_predictions.csv'),('test',test,'predictions.csv')]:
                        y_true = encode_labels(frame,categories)
                        dataset = TextDataset(tokenizer(frame.text.tolist(),truncation=True,max_length=config['protocol']['max_length']),y_true)
                        scores = predict_dataset(model,dataset,tokenizer,batch_size=config['protocol']['eval_batch_size'],
                            num_classes=len(categories),seed=seed,device=device,precision=config['precision']['resolved'])
                        old = pd.read_csv(directory/filename,keep_default_na=False)
                        new_ids = scores.argmax(axis=1)
                        old_ids = old.y_pred_id.to_numpy(dtype=int)
                        # Reject numerical/model drift: every old winner must still have
                        # EXACTLY the top score. Only tied-score ordering may change.
                        if not np.isfinite(scores).all() or not np.array_equal(scores[np.arange(len(scores)),old_ids],scores.max(axis=1)):
                            raise ValueError(f'Re-export differs beyond exact score ties: {directory}/{filename}; originals untouched')
                        changed = int((new_ids != old_ids).sum())
                        changes[split] = dict(changed_predictions=changed,
                            tied_top_rows=int(((scores == scores.max(axis=1,keepdims=True)).sum(axis=1)>1).sum()))
                        predicted = target.write_predictions(frame.text.tolist(),y_true,scores,categories,filename=filename)
                        metrics[split] = classification_metrics(y_true,predicted,categories)
                        if split == 'test':
                            target.write_confusion(y_true,predicted,categories)
                    synchronize(device)
                    extra = {k:v for k,v in original.items() if k not in ('validation','test')}
                    extra.update(prediction_ranking=RANKING,
                        ranking_repair=dict(changes=changes,inference_seconds=time.perf_counter()-started,
                            original_validation_macro_f1=original['validation']['macro_f1'],
                            original_test_macro_f1=original['test']['macro_f1'],
                            note='same saved checkpoint and runtime; exact top-score ties only; no training or model reselection'))
                    target.write_metrics(metrics['validation'],metrics['test'],extra)
                    validate_run(target.path)
                    records.append(dict(directory=directory,staged=target.path,changes=changes))
                    del model
                    gc.collect()
                    if device == 'cuda':
                        torch.cuda.empty_cache()
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        for record in records:
            directory = record['directory']
            backup = directory/'ranking_backup'/stamp
            backup.mkdir(parents=True)
            for name in REPLACED:
                shutil.copy2(directory/name,backup/name)
            try:
                for name in REPLACED:
                    shutil.copy2(record['staged']/name,directory/name)
                validate_run(directory)
            except BaseException:
                for name in REPLACED:
                    shutil.copy2(backup/name,directory/name)
                raise
            print(f'repaired {directory.relative_to(root)}: {record["changes"]}; backup={backup}',flush=True)
    return len(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--setting',nargs='+',choices=SETTINGS,default=['5shot'])
    parser.add_argument('--seed',nargs='+',type=int,default=[42,43,44])
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    print(f'Repaired runs: {repair(args.root,settings=args.setting,seeds=args.seed,device=args.device)}')


if __name__=='__main__':
    main()
