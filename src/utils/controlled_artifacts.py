"""Validate controlled-run evidence before resuming, reporting or selecting a model."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.audit_artifacts import frame_digest
from utils.data import encode_labels, load_splits
from utils.metrics import classification_metrics, confusion_rows


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def file_hashes(directory):
    hashes = {}
    for path in sorted(Path(directory).iterdir()):
        if path.is_file() and path.name != 'manifest.json':
            digest = hashlib.sha256()
            with path.open('rb') as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b''):
                    digest.update(block)
            hashes[path.name] = digest.hexdigest()
    return hashes


def validate_run(directory, *, require_model=False, check_data=True):
    directory = Path(directory)
    config = read_json(directory / 'config.json')
    metrics = read_json(directory / 'metrics.json')
    if read_json(directory / 'status.json').get('status') != 'completed':
        raise ValueError(f'Run is not completed: {directory}')
    if (directory / 'failure.json').exists():
        raise ValueError(f'Completed run still contains a failure record: {directory}')
    if directory.name != f"seed{config['seed']}" or directory.parent.name != config['setting'] or directory.parent.parent.name != config['method']:
        raise ValueError(f'Run identity/path mismatch: {directory}')
    history = read_json(directory / 'training_history.json')
    if metrics['actual_steps'] != config['protocol']['max_steps'] or len(history) != metrics['actual_steps']:
        raise ValueError(f'Incomplete update budget: {directory}')
    if [entry['step'] for entry in history] != list(range(1, len(history) + 1)):
        raise ValueError(f'Invalid history steps: {directory}')
    for entry in history:
        if not all(np.isfinite(entry[k]) for k in ('loss', 'ce', 'supcon', 'grad_norm')):
            raise ValueError(f'Nonfinite history: {directory}')
    candidates = [(e['step'], e['validation_macro_f1']) for e in history if 'validation_macro_f1' in e]
    if not candidates or not all(np.isfinite(f1) for _, f1 in candidates):
        raise ValueError(f'Missing/nonfinite validation selection: {directory}')
    best_step, best_f1 = max(candidates, key=lambda item: item[1])
    if metrics['best_step'] != best_step or not np.isclose(metrics['best_validation_macro_f1'], best_f1, atol=1e-10, rtol=0):
        raise ValueError(f'Checkpoint selection mismatch: {directory}')
    if bool(metrics['smoke_only']) != bool(config['smoke_only']):
        raise ValueError(f'Smoke identity mismatch: {directory}')
    if not read_json(directory / 'env.json').get('packages'):
        raise ValueError(f'Missing environment: {directory}')
    if check_data:
        train, val, test, categories = load_splits(config['setting'], config['seed'])
        if config['smoke_only']:
            val = val.groupby('category', sort=False).head(2).reset_index(drop=True)
            test = test.groupby('category', sort=False).head(2).reset_index(drop=True)
        frames = {'train': train, 'validation': val, 'test': test}
        if config['data_fingerprints'] != {k: frame_digest(v) for k, v in frames.items()}:
            raise ValueError(f'Data fingerprint mismatch: {directory}')
    else:
        categories = config['categories']
        frames = None
    for split, filename in [('validation', 'validation_predictions.csv'), ('test', 'predictions.csv')]:
        pred = pd.read_csv(directory / filename, keep_default_na=False)
        for col in ('y_true_id', 'y_pred_id'):
            ids = pd.to_numeric(pred[col], errors='raise').to_numpy()
            if not np.isfinite(ids).all() or (ids != ids.astype(int)).any() or (ids < 0).any() or (ids >= len(categories)).any():
                raise ValueError(f'Invalid prediction ids: {directory}/{filename}')
        for col, label in [('y_true_id', 'y_true'), ('y_pred_id', 'y_pred')]:
            if pred[label].tolist() != [categories[int(i)] for i in pred[col]]:
                raise ValueError(f'Prediction label mapping mismatch: {directory}/{filename}')
        if len(pred) != config[f'{split}_rows']:
            raise ValueError(f'Prediction row count mismatch: {directory}/{filename}')
        scores = pred[['top1_score', 'top2_score', 'top3_score']].to_numpy(dtype=float)
        if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any() or (np.diff(scores, axis=1) > 1e-7).any():
            raise ValueError(f'Invalid top-k scores: {directory}/{filename}')
        if frames is not None:
            frame = frames[split]
            if pred.text.tolist() != frame.text.tolist() or pred.y_true_id.tolist() != encode_labels(frame, categories).tolist():
                raise ValueError(f'Prediction row order/truth mismatch: {directory}/{filename}')
        recomputed = classification_metrics(pred.y_true_id, pred.y_pred_id, categories)
        for key in ('accuracy', 'macro_f1', 'micro_f1'):
            if not np.isclose(recomputed[key], metrics[split][key], atol=1e-10, rtol=0):
                raise ValueError(f'{split} {key} mismatch: {directory}')
        for category in categories:
            for key, value in recomputed['per_class'][category].items():
                if not np.isclose(value, metrics[split]['per_class'][category][key], atol=1e-10, rtol=0):
                    raise ValueError(f'{split} per-class mismatch: {directory}')
        if split == 'validation' and not np.isclose(recomputed['macro_f1'], best_f1, atol=1e-10, rtol=0):
            raise ValueError(f'Selected validation score mismatch: {directory}')
        if split == 'test':
            matrix = pd.read_csv(directory / 'confusion_matrix.csv', index_col=0)
            expected = pd.DataFrame([row[1:] for row in confusion_rows(pred.y_true_id, pred.y_pred_id, categories)], index=categories, columns=categories)
            if matrix.index.tolist() != categories or matrix.columns.tolist() != categories or not np.array_equal(matrix.to_numpy(), expected.to_numpy()):
                raise ValueError(f'Confusion matrix mismatch: {directory}')
    if require_model:
        model_dir = directory / 'model'
        for name in ('config.json', 'heads.pt', 'categories.json', 'tokenizer_config.json'):
            if not (model_dir / name).exists():
                raise ValueError(f'Missing saved model file: {model_dir / name}')
        if not any((model_dir / name).exists() for name in ('model.safetensors', 'pytorch_model.bin')):
            raise ValueError(f'Missing encoder weights: {model_dir}')
        if config.get('implementation') == 'controlled-delivery-v2':
            if read_json(model_dir / 'manifest.json') != file_hashes(model_dir):
                raise ValueError(f'Saved model checksum mismatch: {directory}')
        if read_json(model_dir / 'categories.json') != categories:
            raise ValueError(f'Saved category order differs: {directory}')
    return config, metrics, categories
