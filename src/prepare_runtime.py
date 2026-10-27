"""Explicit model preparation and hardware/kernel preflight for WSL GPU training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

from config import ROOT
from supcon.joint import model_fingerprints
from utils.artifacts import environment_info
from utils.data import load_splits
from utils.precision import resolve_precision
from utils.seed import select_torch_device


def prepare_model(target, *, download=False, revision='main'):
    target = Path(target)
    if target.exists():
        # An incomplete folder must not be silently overwritten or mixed with a new revision.
        model_fingerprints(target)
        return
    if not download:
        raise FileNotFoundError(f'Model missing: {target}; copy a local snapshot or use --download-model explicitly')
    from huggingface_hub import HfApi, snapshot_download
    repo = 'google-bert/bert-base-uncased'
    commit = HfApi().model_info(repo, revision=revision).sha
    target.parent.mkdir(parents=True,exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='bert-download-',dir=target.parent))
    try:
        snapshot_download(repo_id=repo,revision=commit,local_dir=staging,
            allow_patterns=['config.json','model.safetensors','tokenizer.json','tokenizer_config.json','vocab.txt','special_tokens_map.json'])
        model_fingerprints(staging)
        (staging/'download_info.json').write_text(json.dumps(dict(repo_id=repo,commit=commit),indent=2)+'\n',encoding='utf-8')
        staging.rename(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def preflight(model_path, device='cuda'):
    import torch
    from transformers import AutoTokenizer
    device = select_torch_device(device)
    precision = resolve_precision('auto',device)
    if device == 'cuda':
        capability = torch.cuda.get_device_capability()
        cuda_version = tuple(int(x) for x in (torch.version.cuda or '0.0').split('.')[:2])
        if capability[0] >= 10 and cuda_version < (12,8):
            raise RuntimeError('Blackwell needs a compatible CUDA build; install the documented cu128 wheel')
        if precision['resolved'] != 'bf16':
            raise RuntimeError('The GPU delivery protocol requires BF16 support')
        # Exercise real CUDA forward and backward kernels, rather than trusting is_available().
        x = torch.randn(64,64,device=device,requires_grad=True)
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
            loss = (x@x.T).float().square().mean()
        loss.backward()
        torch.cuda.synchronize()
        if not torch.isfinite(loss) or not torch.isfinite(x.grad).all():
            raise RuntimeError('CUDA BF16 kernel check produced nonfinite values')
    hashes = model_fingerprints(Path(model_path))
    tokenizer = AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    tokenizer('Where is my card?',return_tensors='pt')
    train,val,test,categories = load_splits('5shot',42)
    result = dict(environment=environment_info(),device=device,precision=precision,model_sha256=hashes,
        rows=dict(train=len(train),validation=len(val),test=len(test)),categories=len(categories),
        note='kernel and input preflight only; run the three-arm GPU smoke before formal training')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path',type=Path,default=ROOT/'models/bert-base-uncased')
    parser.add_argument('--download-model',action='store_true')
    parser.add_argument('--revision',default='main',help='resolved to an immutable commit when downloading')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/preflight.json')
    args=parser.parse_args()
    prepare_model(args.model_path,download=args.download_model,revision=args.revision)
    result=preflight(args.model_path,args.device)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
