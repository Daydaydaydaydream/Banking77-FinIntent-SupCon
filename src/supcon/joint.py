"""Controlled end-to-end CE / CE+SupCon / CE+SupCon+HN experiments."""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from analysis.audit_artifacts import frame_digest
from supcon.losses import supervised_contrastive_loss, hard_negative_weight_matrix
from supcon.model import SupConModel, TextDataset, build_loader
from utils.artifacts import RunArtifacts
from utils.data import load_splits, encode_labels
from utils.metrics import classification_metrics
from utils.precision import resolve_precision
from utils.seed import seed_everything, select_torch_device

ARMS = ('ce', 'ce_supcon', 'ce_supcon_hn')


def joint_loss(logits, projected, labels, arm, *, weight=0.1, temperature=0.1, hn_temperature=0.5):
    if arm not in ARMS:
        raise ValueError(f'Unknown arm: {arm}')
    if not math.isfinite(weight) or weight < 0:
        raise ValueError('weight must be finite and nonnegative')
    ce = F.cross_entropy(logits.float(), labels)
    contrastive = ce.new_zeros(())
    if arm != 'ce' and weight > 0:
        features = F.normalize(projected.float(), dim=1)
        weights = hard_negative_weight_matrix(features @ features.t(), labels, hn_temperature) if arm == 'ce_supcon_hn' else None
        contrastive = supervised_contrastive_loss(features, labels, temperature, weights)
    return ce + weight * contrastive, ce, contrastive


def validate_protocol(protocol):
    required = {'protocol','selection','max_steps','eval_steps','learning_rate','weight_decay','shots_per_class','max_length','eval_batch_size','projection_dim','supcon_weight','supcon_temperature','hn_temperature','precision'}
    if set(protocol) != required:
        raise ValueError(f'Protocol keys differ: {set(protocol) ^ required}')
    for key in ('max_steps','eval_steps','shots_per_class','max_length','eval_batch_size','projection_dim'):
        if type(protocol[key]) is not int or protocol[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if protocol['shots_per_class'] < 2: raise ValueError('shots_per_class must be >= 2')
    for key in ('learning_rate','supcon_weight','supcon_temperature','hn_temperature'):
        if not math.isfinite(protocol[key]) or protocol[key] <= 0: raise ValueError(f'{key} must be finite and positive')
    if not math.isfinite(protocol['weight_decay']) or protocol['weight_decay'] < 0: raise ValueError('weight_decay must be finite and nonnegative')


def model_fingerprints(model_path: Path):
    if not model_path.is_dir(): raise ValueError('Controlled experiments require a local model snapshot')
    hashes = {}
    for path in sorted(model_path.iterdir()):
        if path.is_file() and path.suffix in ('.json','.txt','.bin','.safetensors'):
            digest=hashlib.sha256()
            with path.open('rb') as handle:
                for block in iter(lambda:handle.read(1024*1024),b''): digest.update(block)
            hashes[path.name]=digest.hexdigest()
    if not any(name.endswith(('.safetensors','.bin')) for name in hashes): raise ValueError('Local model weights missing')
    return hashes


def train_loop(model, optimizer, loader_factory, evaluate, *, arm, protocol, device, resolved_precision):
    """Fixed update budget; validation alone selects the complete model state."""
    step, epoch, best_f1, best_step, best_state = 0, 0, -1., 0, None
    history = []
    started=time.perf_counter()
    while step < protocol['max_steps']:
        loader=loader_factory(epoch)
        if len(loader)==0: raise ValueError('Empty training loader')
        for batch in loader:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            labels=batch['labels'].to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=resolved_precision=='bf16'):
                projected, logits=model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device))
            loss,ce,supcon=joint_loss(logits,projected,labels,arm,weight=protocol['supcon_weight'],temperature=protocol['supcon_temperature'],hn_temperature=protocol['hn_temperature'])
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss; run invalid')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm): raise FloatingPointError('Nonfinite gradient; run invalid')
            optimizer.step()
            step+=1
            entry=dict(step=step,epoch=epoch,loss=float(loss.detach()),ce=float(ce.detach()),supcon=float(supcon.detach()),grad_norm=float(norm))
            if step % protocol['eval_steps']==0 or step==protocol['max_steps']:
                f1=evaluate(model)
                if not math.isfinite(f1): raise FloatingPointError('Nonfinite validation metric')
                entry['validation_macro_f1']=f1
                if f1>best_f1:
                    best_f1,best_step=f1,step
                    best_state={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
            history.append(entry)
            if step==protocol['max_steps']: break
        epoch+=1
    elapsed=time.perf_counter()-started
    model.load_state_dict(best_state)
    return history, dict(actual_steps=step,best_step=best_step,best_validation_macro_f1=best_f1,train_seconds=elapsed)


def run(arm, setting, seed, *, protocol, model_path, model_hashes, root, device='auto', save_model=True, smoke=False):
    validate_protocol(protocol)
    if arm not in ARMS: raise ValueError('Unknown arm')
    artifacts=RunArtifacts(arm,setting,seed,root=Path(root))
    if any(artifacts.path.iterdir()): raise FileExistsError(f'Refusing to overwrite {artifacts.path}')
    artifacts.write_status('running')
    try:
        seed_info=seed_everything(seed)
        device=select_torch_device(device)
        precision=resolve_precision(protocol['precision'],device)
        if precision['resolved']=='fp16': raise ValueError('Controlled runner supports fp32/bf16 only')
        train,val,test,categories=load_splits(setting,seed)
        if smoke:
            val=val.groupby('category',sort=False).head(2).reset_index(drop=True)
            test=test.groupby('category',sort=False).head(2).reset_index(drop=True)
        fingerprints={key:frame_digest(frame) for key,frame in [('train',train),('validation',val),('test',test)]}
        config=dict(method=arm,setting=setting,seed=seed,protocol=protocol,device=device,precision=precision,seed_info=seed_info,
                    model_name=str(model_path),model_sha256=model_hashes,data_fingerprints=fingerprints,
                    train_rows=len(train),validation_rows=len(val),test_rows=len(test),smoke_only=smoke,
                    effective_batch_size=len(categories)*protocol['shots_per_class'],loss_implementation='negative-mean-one-detached-logsumexp-v2')
        artifacts.write_config(config)
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
        frames=[train,val,test]
        datasets=[TextDataset(tokenizer(frame.text.tolist(),truncation=True,max_length=protocol['max_length']),encode_labels(frame,categories)) for frame in frames]
        model=SupConModel(str(model_path),len(categories),projection_dim=protocol['projection_dim'],local_files_only=True).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=protocol['learning_rate'],weight_decay=protocol['weight_decay'])
        def loader_factory(epoch):
            # All arms see the same batches, with deterministic reshuffling each epoch.
            return build_loader(datasets[0],tokenizer,batch_size=config['effective_batch_size'],num_classes=len(categories),seed=seed+epoch,shuffle=True,shots_per_class=protocol['shots_per_class'])
        def predict(dataset):
            model.eval()
            outputs=[]
            loader=build_loader(dataset,tokenizer,batch_size=protocol['eval_batch_size'],num_classes=len(categories),seed=seed,shuffle=False)
            with torch.no_grad():
                for batch in loader:
                    pooled=model.embed(batch['input_ids'].to(device),batch['attention_mask'].to(device))
                    outputs.append(torch.softmax(model.classify(pooled).float(),dim=1).cpu().numpy())
            return np.concatenate(outputs)
        y_val=encode_labels(val,categories)
        history,extra=train_loop(model,optimizer,loader_factory,lambda _:classification_metrics(y_val,predict(datasets[1]).argmax(1),categories)['macro_f1'],arm=arm,protocol=protocol,device=device,resolved_precision=precision['resolved'])
        artifacts.write_history(history)
        val_scores=predict(datasets[1])
        val_pred=artifacts.write_predictions(val.text.tolist(),y_val,val_scores,categories,filename='validation_predictions.csv')
        val_metrics=classification_metrics(y_val,val_pred,categories)
        # Test inference starts only after validation-selected state is restored.
        started=time.perf_counter()
        scores=predict(datasets[2])
        extra['test_inference_seconds']=time.perf_counter()-started
        y_test=encode_labels(test,categories)
        predicted=artifacts.write_predictions(test.text.tolist(),y_test,scores,categories)
        artifacts.write_confusion(y_test,predicted,categories)
        metrics=classification_metrics(y_test,predicted,categories)
        if save_model:
            model.encoder.save_pretrained(artifacts.model_dir)
            torch.save({'projection':model.projection.state_dict(),'classifier':model.classifier.state_dict()},artifacts.model_dir/'heads.pt')
            tokenizer.save_pretrained(artifacts.model_dir)
            model.encoder.config.save_pretrained(artifacts.model_dir)
            (artifacts.model_dir/'categories.json').write_text(json.dumps(categories,indent=2)+'\n')
        artifacts.write_metrics(val_metrics,metrics,{**extra,'smoke_only':smoke})
        artifacts.write_status('completed')
        return dict(method=arm,setting=setting,seed=seed,test_macro_f1=metrics['macro_f1'],**extra)
    except Exception as exc:
        artifacts.write_failure(exc,'joint training')
        raise


def load_saved_model(directory: Path, device='cpu'):
    """Restore a self-contained selected encoder and both task heads."""
    config=json.loads((directory/'config.json').read_text())
    categories=json.loads((directory/'model/categories.json').read_text())
    model=SupConModel(str(directory/'model'),len(categories),projection_dim=config['protocol']['projection_dim'],local_files_only=True)
    heads=torch.load(directory/'model/heads.pt',map_location='cpu',weights_only=True)
    model.projection.load_state_dict(heads['projection'])
    model.classifier.load_state_dict(heads['classifier'])
    return model.to(device).eval(), categories
