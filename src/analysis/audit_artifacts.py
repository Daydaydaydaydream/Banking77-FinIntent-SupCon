"""Audit historical runs without inventing missing historical provenance."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import ROOT, RUNS_DIR, SETTINGS
from data.prepare_data import create_split, normalize_text
from utils.data import load_splits, encode_labels
from utils.metrics import classification_metrics


def frame_digest(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame.to_csv(index=False, lineterminator="\n").encode()).hexdigest()


def check_confusion(matrix: pd.DataFrame, metrics: dict, categories: list[str]) -> list[str]:
    errors = []
    if matrix.index.tolist() != categories or matrix.columns.tolist() != categories:
        return ["confusion label order mismatch"]
    values = matrix.to_numpy()
    if not np.isfinite(values).all() or (values < 0).any() or (values != values.astype(int)).any():
        return ["invalid confusion counts"]
    if not (values.sum(axis=1) == 40).all():
        errors.append("test support is not 40 per class")
    actual, predicted = np.nonzero(values)
    counts = values[actual, predicted].astype(int)
    recalculated = classification_metrics(np.repeat(actual, counts), np.repeat(predicted, counts), categories)
    for key in ("accuracy", "macro_f1", "micro_f1"):
        if not np.isclose(recalculated[key], metrics[key], atol=1e-10, rtol=0):
            errors.append(f"test {key} mismatch")
    for category in categories:
        for key, value in recalculated['per_class'][category].items():
            if not np.isclose(value, metrics['per_class'][category][key], atol=1e-10, rtol=0):
                errors.append(f"test per_class {category}/{key} mismatch")
    return errors


def audit(runs_dir: Path = RUNS_DIR) -> dict:
    # Recreate deterministic derived data, never rewrite historical run artifacts.
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as temporary:
        manifest = create_split(output_dir=Path(temporary))
        frames = {name: pd.read_csv(Path(temporary) / f'{name}.csv') for name in ('train', 'val', 'test')}
    from config import DATASET_DIR
    split_errors = []
    for name, frame in frames.items():
        saved = DATASET_DIR / f'{name}.csv'
        if saved.exists() and not frame.equals(pd.read_csv(saved)):
            split_errors.append(f'local {name} differs from deterministic reconstruction')
    norm = {name: set(frame.text.map(normalize_text)) for name, frame in frames.items()}
    if norm['train'] & norm['val']:
        split_errors.append('normalized train/validation overlap')
    categories = json.loads((ROOT/'data/banking_data/categories.json').read_text())
    from utils.data import sample_few_shot
    runs = []
    for method in ('tfidf_svm', 'bert', 'setfit', 'supcon', 'supcon_hn'):
        for setting in SETTINGS:
            for seed in (42,43,44):
                directory = runs_dir/method/setting/f'seed{seed}'
                errors, limitations = [], []
                train = frames['train'] if setting == 'full' else sample_few_shot(frames['train'], int(setting[:-4]), seed)
                try:
                    config = json.loads((directory/'config.json').read_text())
                    metrics = json.loads((directory/'metrics.json').read_text())
                    env = json.loads((directory/'env.json').read_text())
                    status = json.loads((directory/'status.json').read_text())
                    for key, expected in dict(method=method, setting=setting, seed=seed, train_rows=len(train), validation_rows=1003, test_rows=3080).items():
                        if config.get(key) != expected:
                            errors.append(f'config {key} mismatch: {config.get(key)!r}')
                    if status.get('status') != 'completed': errors.append('status not completed')
                    if not env.get('packages'): errors.append('missing environment packages')
                    matrix = pd.read_csv(directory/'confusion_matrix.csv', index_col=0)
                    errors += check_confusion(matrix, metrics['test'], categories)
                    val = metrics['validation']['per_class']
                    supports = frames['val'].category.value_counts()
                    if any(val[c]['support'] != int(supports[c]) for c in categories): errors.append('validation support mismatch')
                    if not np.isclose(np.mean([val[c]['f1'] for c in categories]),metrics['validation']['macro_f1'],atol=1e-10,rtol=0): errors.append('validation macro F1 mismatch')
                    predictions = directory/'predictions.csv'
                    if predictions.exists():
                        pred = pd.read_csv(predictions)
                        if pred.text.tolist()!=frames['test'].text.tolist() or pred.y_true.tolist()!=frames['test'].category.tolist(): errors.append('prediction test row mismatch')
                        if pred.y_pred.tolist()!=[categories[i] for i in pred.y_pred_id]: errors.append('prediction label mismatch')
                        if pred.y_true_id.tolist()!=encode_labels(frames['test'],categories).tolist(): errors.append('prediction truth id mismatch')
                        from utils.metrics import confusion_rows
                        if confusion_rows(pred.y_true_id,pred.y_pred_id,categories)!=[[c,*map(int,matrix.loc[c])] for c in categories]: errors.append('prediction/confusion mismatch')
                    else: limitations.append('historical test predictions missing')
                    if not (directory/'validation_predictions.csv').exists(): limitations.append('historical validation predictions missing')
                    if 'data_fingerprints' not in config: limitations.append('historical training data fingerprint missing; reconstructed sampling is not proof of historical usage')
                except (OSError, ValueError, KeyError, IndexError) as exc:
                    errors.append(str(exc))
                runs.append(dict(run=f'{method}/{setting}/seed{seed}',errors=errors,limitations=limitations,reconstructed_train_sha256=frame_digest(train)))
    return dict(expected_runs=60, audited_runs=len(runs), errors=sum(len(r['errors']) for r in runs)+len(split_errors),
                runs_with_missing_test_predictions=sum('historical test predictions missing' in r['limitations'] for r in runs),
                split_errors=split_errors, reconstructed_split=manifest['sizes'], source_sha256=manifest['source'],
                split_sha256={k:frame_digest(v) for k,v in frames.items()},
                normalized_overlap={f'{a}/{b}':len(norm[a]&norm[b]) for a,b in [('train','val'),('train','test'),('val','test')]},
                runs=runs)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/results/artifact_audit.json')
    args=parser.parse_args()
    result=audit()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='runs'},ensure_ascii=False,indent=2))
    raise SystemExit(1 if result['errors'] else 0)

if __name__=='__main__': main()
