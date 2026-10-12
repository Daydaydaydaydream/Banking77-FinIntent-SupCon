"""Run a frozen, shared protocol for the three controlled loss arms."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from config import ROOT, SETTINGS, BERT_MODEL
from supcon.joint import ARMS, model_fingerprints, run, validate_protocol
from analysis.audit_artifacts import frame_digest
from utils.data import load_splits


def summarise(root):
    rows=[]
    for file in sorted(root.glob('*/*/seed*/metrics.json')):
        directory=file.parent
        config=json.loads((directory/'config.json').read_text())
        if config['smoke_only'] or json.loads((directory/'status.json').read_text())['status']!='completed': continue
        metrics=json.loads(file.read_text())
        rows.append(dict(method=config['method'],setting=config['setting'],seed=config['seed'],macro_f1=metrics['test']['macro_f1'],train_seconds=metrics['train_seconds']))
    if not rows: return
    frame=pd.DataFrame(rows)
    frame.to_csv(root/'runs.csv',index=False)
    frame.groupby(['method','setting']).agg(seeds=('seed','count'),macro_f1_mean=('macro_f1','mean'),macro_f1_std=('macro_f1','std'),train_seconds_mean=('train_seconds','mean')).to_csv(root/'summary.csv')
    paired=[]
    for baseline,treatment in [('ce','ce_supcon'),('ce_supcon','ce_supcon_hn')]:
        pairs=frame[frame.method==baseline].merge(frame[frame.method==treatment],on=['setting','seed'],suffixes=('_base','_treat'))
        pairs['delta']=pairs.macro_f1_treat-pairs.macro_f1_base
        for setting,group in pairs.groupby('setting'):
            paired.append(dict(baseline=baseline,treatment=treatment,setting=setting,seeds=len(group),mean_delta=group.delta.mean(),std_delta=group.delta.std() if len(group)>1 else None,worst_delta=group.delta.min()))
    (root/'paired_comparisons.json').write_text(json.dumps(paired,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=ROOT/'configs/controlled_v1.json')
    parser.add_argument('--method',nargs='+',choices=ARMS,default=list(ARMS))
    parser.add_argument('--setting',nargs='+',choices=SETTINGS,default=list(SETTINGS))
    parser.add_argument('--seed',type=int,nargs='+',default=[42,43,44])
    parser.add_argument('--model-path',type=Path,default=Path(BERT_MODEL))
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/controlled_runs')
    parser.add_argument('--device',choices=['auto','cpu','mps','cuda'],default='auto')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--no-save-model',action='store_true')
    parser.add_argument('--smoke',action='store_true',help='one update per arm, 5-shot seed42, small evaluation subsets; never formal results')
    args=parser.parse_args()
    import torch
    if args.threads<=0: parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    protocol=json.loads(args.protocol.read_text())
    validate_protocol(protocol)
    if args.smoke:
        protocol={**protocol,'max_steps':1,'eval_steps':1}
        args.setting,args.seed=['5shot'],[42]
        args.output_dir=args.output_dir/'smoke'
    hashes=model_fingerprints(args.model_path)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    frozen=args.output_dir/'protocol.json'
    identity=dict(protocol=protocol,model_sha256=hashes,smoke_only=args.smoke)
    if frozen.exists() and json.loads(frozen.read_text())!=identity:
        raise ValueError('Output root already has a different protocol/model. Choose a new output directory.')
    frozen.write_text(json.dumps(identity,indent=2)+'\n')
    failures=[]
    for setting in dict.fromkeys(args.setting):
        for seed in dict.fromkeys(args.seed):
            for arm in dict.fromkeys(args.method):
                directory=args.output_dir/arm/setting/f'seed{seed}'
                if (directory/'status.json').exists() and json.loads((directory/'status.json').read_text()).get('status')=='completed':
                    config=json.loads((directory/'config.json').read_text())
                    train,val,test,_=load_splits(setting,seed)
                    if args.smoke:
                        val=val.groupby('category',sort=False).head(2).reset_index(drop=True)
                        test=test.groupby('category',sort=False).head(2).reset_index(drop=True)
                    expected={k:frame_digest(f) for k,f in [('train',train),('validation',val),('test',test)]}
                    if config['protocol']!=protocol or config['model_sha256']!=hashes or config['data_fingerprints']!=expected:
                        raise ValueError(f'Completed run does not match requested protocol/data: {directory}')
                    for name in ['metrics.json','predictions.csv','validation_predictions.csv','training_history.json','confusion_matrix.csv']:
                        if not (directory/name).exists(): raise ValueError(f'Missing completed artifact: {directory/name}')
                    print(f'skip verified completed run {arm}/{setting}/{seed}',flush=True)
                    continue
                print(f'run {arm}/{setting}/{seed}',flush=True)
                try:
                    result=run(arm,setting,seed,protocol=protocol,model_path=args.model_path,model_hashes=hashes,root=args.output_dir,device=args.device,save_model=not args.no_save_model,smoke=args.smoke)
                    print(json.dumps(result),flush=True)
                except Exception as exc:
                    failures.append(dict(method=arm,setting=setting,seed=seed,error=str(exc)))
                    print(f'FAILED: {exc}',flush=True)
    summarise(args.output_dir)
    if failures:
        print(json.dumps(failures,indent=2))
        raise SystemExit(1)

if __name__=='__main__': main()
