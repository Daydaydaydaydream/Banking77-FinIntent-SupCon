"""Audit a complete controlled matrix and produce an evidence-based delivery report."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.confusion_pairs import declared_pairs
from config import SETTINGS
from supcon.joint import ARMS
from utils.controlled_artifacts import read_json, validate_run

COMPARISONS = [('ce', 'ce_supcon'), ('ce_supcon', 'ce_supcon_hn')]


def build_report(root, *, settings=('5shot',), seeds=(42,43,44), output=None, select_model=True):
    root = Path(root).resolve()
    output = Path(output).resolve() if output else root/'delivery'
    if 'smoke' in root.parts:
        raise ValueError('Smoke results cannot be used for a formal delivery report')
    settings, seeds = list(dict.fromkeys(settings)), sorted(set(seeds))
    if not settings or not seeds:
        raise ValueError('At least one setting and seed is required')
    identity = read_json(root/'protocol.json')
    if identity['smoke_only']:
        raise ValueError('Smoke protocol cannot be used for a formal report')
    runs, predictions, matrices = {}, {}, {}
    rows = []
    for setting in settings:
        for seed in seeds:
            reference = None
            for arm in ARMS:
                directory = root/arm/setting/f'seed{seed}'
                if not (directory/'metrics.json').exists():
                    raise ValueError(f'Missing required formal run: {directory}')
                config, metrics, categories = validate_run(directory)
                if config['smoke_only']:
                    raise ValueError(f'Smoke run in formal matrix: {directory}')
                for key in ('protocol', 'model_sha256', 'implementation'):
                    if config.get(key) != identity.get(key):
                        raise ValueError(f'{key} differs from frozen root: {directory}')
                if config['device'] != identity.get('device') or config['effective_batch_size'] != len(categories)*config['protocol']['shots_per_class']:
                    raise ValueError(f'Device/batch differs from protocol: {directory}')
                if read_json(directory/'env.json') != identity.get('environment'):
                    raise ValueError(f'Runtime differs across matrix: {directory}')
                signature = (config['data_fingerprints'], categories, config['effective_batch_size'], config['precision'], config['device'])
                if reference is not None and reference != signature:
                    raise ValueError(f'Unpaired data/training conditions: {directory}')
                reference = signature
                runs[arm,setting,seed] = (directory, config, metrics)
                predictions[arm,setting,seed] = pd.read_csv(directory/'predictions.csv', keep_default_na=False)
                matrices[arm,setting,seed] = pd.read_csv(directory/'confusion_matrix.csv', index_col=0)
                pred = predictions[arm,setting,seed]
                rows.append(dict(method=arm, setting=setting, seed=seed,
                    validation_macro_f1=metrics['validation']['macro_f1'], test_macro_f1=metrics['test']['macro_f1'],
                    accuracy=metrics['test']['accuracy'], train_seconds=metrics['train_seconds'],
                    inference_seconds=metrics.get('test_inference_seconds'),
                    examples_per_second=metrics.get('test_examples_per_second'),
                    peak_allocated_gib=(metrics['cuda_peak_allocated_bytes']/1024**3 if metrics.get('cuda_peak_allocated_bytes') is not None else None),
                    actual_steps=metrics['actual_steps'], best_step=metrics['best_step'],
                    predicted_class_count=pred.y_pred.nunique(),
                    zero_f1_classes=sum(v['f1']==0 for v in metrics['test']['per_class'].values())))
    frame = pd.DataFrame(rows)
    paired_rows, class_rows, pair_rows, cases = [], [], [], []
    for baseline, treatment in COMPARISONS:
        for setting in settings:
            for seed in seeds:
                base_dir, _, bm = runs[baseline,setting,seed]
                _, _, tm = runs[treatment,setting,seed]
                base, treat = predictions[baseline,setting,seed], predictions[treatment,setting,seed]
                if not base[['text','y_true']].equals(treat[['text','y_true']]):
                    raise ValueError('Paired prediction rows differ')
                paired_rows.append(dict(baseline=baseline,treatment=treatment,setting=setting,seed=seed,
                    delta=tm['test']['macro_f1']-bm['test']['macro_f1']))
                for category in categories:
                    class_rows.append(dict(baseline=baseline,treatment=treatment,setting=setting,seed=seed,category=category,
                        base_f1=bm['test']['per_class'][category]['f1'], treatment_f1=tm['test']['per_class'][category]['f1'],
                        delta=tm['test']['per_class'][category]['f1']-bm['test']['per_class'][category]['f1']))
                for group, a, b in declared_pairs():
                    if a not in categories or b not in categories:
                        continue
                    bc, tc = matrices[baseline,setting,seed], matrices[treatment,setting,seed]
                    before, after = int(bc.loc[a,b]+bc.loc[b,a]), int(tc.loc[a,b]+tc.loc[b,a])
                    pair_rows.append(dict(baseline=baseline,treatment=treatment,setting=setting,seed=seed,group=group,
                        label_a=a,label_b=b,base_errors=before,treatment_errors=after,delta=after-before))
                # Fixed seed, first rows in dataset order, and explicit strata: no cherry-picked anecdotes.
                if seed == seeds[0]:
                    b_ok, t_ok = base.y_pred.eq(base.y_true), treat.y_pred.eq(treat.y_true)
                    for kind, mask in [('improved',~b_ok&t_ok), ('worsened',b_ok&~t_ok), ('common_error',~b_ok&~t_ok)]:
                        for idx in base.index[mask][:2]:
                            cases.append(dict(setting=setting,seed=seed,baseline=baseline,treatment=treatment,kind=kind,
                                row_index=int(idx),text=base.loc[idx,'text'],truth=base.loc[idx,'y_true'],
                                base_prediction=base.loc[idx,'y_pred'],treatment_prediction=treat.loc[idx,'y_pred']))
    paired = pd.DataFrame(paired_rows)
    paired_summary = []
    for (baseline,treatment,setting), group in paired.groupby(['baseline','treatment','setting'],sort=False):
        paired_summary.append(dict(baseline=baseline,treatment=treatment,setting=setting,seeds=len(group),
            mean_delta=float(group.delta.mean()),std_delta=float(group.delta.std()) if len(group)>1 else None,
            positive_seeds=int((group.delta>0).sum()),negative_seeds=int((group.delta<0).sum()),
            per_seed=group[['seed','delta']].to_dict('records')))
    summary = frame.groupby(['method','setting'],sort=False).agg(
        seeds=('seed','count'), macro_f1_mean=('test_macro_f1','mean'),macro_f1_std=('test_macro_f1','std'),
        validation_f1_mean=('validation_macro_f1','mean'),train_seconds_mean=('train_seconds','mean'),
        examples_per_second_mean=('examples_per_second','mean'),peak_allocated_gib_max=('peak_allocated_gib','max')).reset_index()
    selected = None
    if select_model:
        # Select a METHOD by mean validation F1, then use a predeclared seed for the demo.
        # Do not choose the most fortunate test seed/checkpoint.
        demo_setting = settings[0]
        ranking = frame[frame.setting==demo_setting].groupby('method').validation_macro_f1.mean()
        method = sorted(ARMS, key=lambda a:(-ranking[a],ARMS.index(a)))[0]
        directory, config, metrics = runs[method,demo_setting,seeds[0]]
        validate_run(directory, require_model=True)
        selected = dict(run_dir=os.path.relpath(directory,output),method=method,setting=demo_setting,seed=seeds[0],
            rule='method with highest mean validation Macro-F1; fixed smallest requested seed; ties use CE, SupCon, HN order',
            validation_mean=float(ranking[method]),test_used_for_selection=False,
            note='one demo checkpoint, not an ensemble or the reported multi-seed mean')
    output.mkdir(parents=True,exist_ok=True)
    frame.to_csv(output/'runs.csv',index=False)
    summary.to_csv(output/'summary.csv',index=False)
    paired.to_csv(output/'paired_deltas.csv',index=False)
    pd.DataFrame(class_rows).to_csv(output/'per_class_deltas.csv',index=False)
    pd.DataFrame(pair_rows).to_csv(output/'confusion_pairs.csv',index=False)
    pair_summary = []
    if pair_rows:
        pairs = pd.DataFrame(pair_rows)
        for keys, group in pairs.groupby(['baseline','treatment','setting','group','label_a','label_b'],sort=False):
            pair_summary.append(dict(zip(['baseline','treatment','setting','group','label_a','label_b'],keys),
                mean_error_delta=float(group.delta.mean()),all_seeds_improved=bool((group.delta<0).all()),
                all_seeds_worsened=bool((group.delta>0).all())))
    pd.DataFrame(pair_summary).to_csv(output/'confusion_pair_summary.csv',index=False)
    (output/'paired_comparisons.json').write_text(json.dumps(paired_summary,indent=2)+'\n',encoding='utf-8')
    (output/'error_cases.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if selected:
        (output/'selected_model.json').write_text(json.dumps(selected,indent=2)+'\n',encoding='utf-8')
    elif (output/'selected_model.json').exists():
        (output/'selected_model.json').unlink()
    lines = ['# 受控实验交付报告','',f'复核通过：{len(frame)} 组；设置：{", ".join(settings)}；seeds：{seeds}。',
        '只包含本次完整正式矩阵；历史 HN 与 smoke 不参与。种子数量有限，不作普遍性结论。','',
        '## 效果与成本','',
        '| 方法 | 设置 | Test Macro-F1 mean ± std | 训练及周期验证秒数均值 | 测试条/秒均值 | 峰值 allocated GiB |',
        '|---|---|---:|---:|---:|---:|']
    def fmt(value, digits=4):
        return f'{value:.{digits}f}' if pd.notna(value) else 'N/A'
    for r in summary.itertuples():
        lines.append(f'| {r.method} | {r.setting} | {fmt(r.macro_f1_mean)} ± {fmt(r.macro_f1_std)} | {fmt(r.train_seconds_mean,1)} | {fmt(r.examples_per_second_mean,1)} | {fmt(r.peak_allocated_gib_max,2)} |')
    lines += ['', '## 配对差值','']
    for item in paired_summary:
        lines.append(f'- {item["setting"]} {item["treatment"]} − {item["baseline"]}：均值 {item["mean_delta"]:+.4f}；正向 {item["positive_seeds"]}/{item["seeds"]}，负向 {item["negative_seeds"]}/{item["seeds"]}。')
    lines += ['', '## 证据与局限','',
        '- 每组配置、数据指纹、逐条预测、逐类指标、混淆矩阵和最佳验证步数均已复核。',
        '- train_seconds 含训练、周期验证和最佳状态 CPU 拷贝；不含模型加载、分词及最终保存。测试吞吐经过预热，包含数据装载、前向与分数回传，不含分词。详见各组 metrics.json。',
        '- 峰值显存为该次训练及评估过程中 PyTorch 分配的峰值，不等于系统总显存占用；不同硬件/精度的时间不直接比较。',
        '- few-shot 仅指训练标签，另用 1,003 条有标签验证数据。已有测试结果已查看；配置只按预声明协议及验证规则选择。官方数据存在已披露的跨 train/test 与 validation/test 重复文本。',
        '- 29 对预定义混淆的逐种子及聚合结果见 confusion_pairs.csv / confusion_pair_summary.csv；改善对应错误数差值为负。',
        '- error_cases.json 按最小请求种子、数据原顺序，每对方法各取改善/恶化/共同失败最多两条；缺少某层不补造。案例只用于解释，不代替总体统计。',
        '- 这三组比较隔离当前协议下损失的增量，不宣称各方法均已充分调优；负结果同样有效。']
    if selected:
        lines += ['', '## 演示模型','',f'按验证集选中方法 {selected["method"]}，固定 seed {selected["seed"]}，设置 {selected["setting"]}。',
                  'selected_model.json 指向可离线加载的模型；演示模型不是多种子均值，也不是按测试分数挑选的最佳 seed。']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--setting',nargs='+',choices=SETTINGS,default=['5shot'])
    parser.add_argument('--seed',nargs='+',type=int,default=[42,43,44])
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--no-select-model',action='store_true',help='audit/report only; do not export a demo selection')
    args=parser.parse_args()
    print(build_report(args.root,settings=args.setting,seeds=args.seed,output=args.output_dir,select_model=not args.no_select_model))


if __name__=='__main__':
    main()
