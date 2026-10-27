"""Meaningful offline integration: train, audit, select by validation and reload."""
import contextlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from analysis.controlled_report import build_report
from predict import IntentPredictor
from supcon.joint import ARMS, IMPLEMENTATION, model_fingerprints, run
from utils.artifacts import environment_info
from utils.controlled_artifacts import read_json, validate_run


class DeliveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import BertConfig, BertModel, BertTokenizerFast
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.snapshot = cls.root/'snapshot'
        cls.snapshot.mkdir()
        (cls.snapshot/'vocab.txt').write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nrefund\ncard\nwhere\nwhy\n')
        BertTokenizerFast(vocab_file=str(cls.snapshot/'vocab.txt')).save_pretrained(cls.snapshot)
        BertModel(BertConfig(vocab_size=9,hidden_size=8,num_hidden_layers=1,num_attention_heads=2,intermediate_size=16)).save_pretrained(cls.snapshot)
        cls.categories = ['refund','card','where']
        cls.frame = pd.DataFrame(dict(text=['refund','why refund','card','why card','where','where card'],
                                      category=['refund','refund','card','card','where','where']))
        cls.splits = (cls.frame,cls.frame,cls.frame,cls.categories)
        cls.protocol = read_json(Path(__file__).resolve().parents[1]/'configs/controlled_gpu_v2.json')
        cls.protocol.update(max_steps=2,eval_steps=1,projection_dim=4,eval_batch_size=4,max_length=8,precision='auto')
        cls.hashes = model_fingerprints(cls.snapshot)
        cls.runs = cls.root/'runs'
        cls.runs.mkdir()
        cls.identity = dict(protocol=cls.protocol,model_sha256=cls.hashes,smoke_only=False,
                            implementation=IMPLEMENTATION,device='cpu',environment=environment_info())
        (cls.runs/'protocol.json').write_text(json.dumps(cls.identity))
        with patch('supcon.joint.load_splits',return_value=cls.splits), contextlib.redirect_stdout(io.StringIO()):
            for seed in (42,43,44):
                for arm in ARMS:
                    run(arm,'5shot',seed,protocol=cls.protocol,model_path=cls.snapshot,model_hashes=cls.hashes,
                        root=cls.runs,device='cpu')

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_complete_report_and_validation_only_selection(self):
        with patch('utils.controlled_artifacts.load_splits',return_value=self.splits):
            out = build_report(self.runs)
        table = pd.read_csv(out/'runs.csv')
        self.assertEqual(len(table),9)
        selected = read_json(out/'selected_model.json')
        mean_val = table.groupby('method').validation_macro_f1.mean()
        expected = sorted(ARMS,key=lambda a:(-mean_val[a],ARMS.index(a)))[0]
        self.assertEqual(selected['method'],expected)
        self.assertEqual(selected['seed'],42)
        self.assertFalse(selected['test_used_for_selection'])
        self.assertTrue((out/selected['run_dir']/'model/heads.pt').exists())
        self.assertEqual(len(pd.read_csv(out/'paired_deltas.csv')),6)
        self.assertIn('固定 seed 42',(out/'REPORT.md').read_text())

    def test_predict_matches_saved_scores_and_handles_long_text(self):
        directory = self.runs/'ce/5shot/seed42'
        predictor = IntentPredictor(directory,device='cpu')
        result = predictor.predict(self.frame.text.tolist(),batch_size=2)
        saved = pd.read_csv(directory/'predictions.csv')
        self.assertEqual([r['label'] for r in result['predictions']],saved.y_pred.tolist())
        self.assertTrue(np.allclose([r['top_k'][0]['score'] for r in result['predictions']],saved.top1_score,atol=1e-6))
        self.assertTrue(predictor.predict(['card '*30])['predictions'][0]['truncated'])
        for bad in ([],[''],['   '],['card',None]):
            with self.assertRaises(ValueError): predictor.predict(bad)
        with self.assertRaises(ValueError): predictor.predict(['card'],top_k=4)

    def test_audit_detects_metric_corruption(self):
        directory = self.runs/'ce/5shot/seed42'
        path = directory/'metrics.json'
        original = path.read_text()
        try:
            data = json.loads(original)
            data['test']['macro_f1'] += .1
            path.write_text(json.dumps(data))
            with patch('utils.controlled_artifacts.load_splits',return_value=self.splits):
                with self.assertRaisesRegex(ValueError,'macro_f1 mismatch'): validate_run(directory)
        finally:
            path.write_text(original)

    def test_model_checksum_detects_corruption(self):
        path = self.runs/'ce/5shot/seed42/model/categories.json'
        original = path.read_text()
        try:
            path.write_text(json.dumps(list(reversed(self.categories))))
            with self.assertRaisesRegex(ValueError,'checksum mismatch'):
                IntentPredictor(path.parent.parent,device='cpu')
        finally:
            path.write_text(original)

    def test_report_refuses_incomplete_or_smoke_matrix(self):
        path = self.runs/'ce/5shot/seed44/metrics.json'
        original = path.read_text()
        try:
            path.unlink()
            with patch('utils.controlled_artifacts.load_splits',return_value=self.splits):
                with self.assertRaisesRegex(ValueError,'Missing required formal run'): build_report(self.runs)
        finally:
            path.write_text(original)
        with self.assertRaisesRegex(ValueError,'Smoke'):
            build_report(self.root/'smoke')

    def test_runner_resumes_verified_runs_and_archives_incomplete(self):
        import run_controlled
        target = self.root/'resume-runs'
        shutil.copytree(self.runs,target)
        directory = target/'ce/5shot/seed42'
        (directory/'status.json').write_text(json.dumps(dict(status='failed')))
        args = ['run_controlled','--protocol',str(self.root/'protocol.json'),'--model-path',str(self.snapshot),
                '--output-dir',str(target),'--device','cpu','--method','ce','--setting','5shot','--seed','42']
        (self.root/'protocol.json').write_text(json.dumps(self.protocol))
        with patch('sys.argv',args), patch('utils.controlled_artifacts.load_splits',return_value=self.splits):
            with self.assertRaisesRegex(ValueError,'Incomplete run'): run_controlled.main()
        with patch('sys.argv',args+['--retry-incomplete']), patch('supcon.joint.load_splits',return_value=self.splits), \
                patch('utils.controlled_artifacts.load_splits',return_value=self.splits), contextlib.redirect_stdout(io.StringIO()):
            run_controlled.main()
        self.assertEqual(read_json(directory/'status.json')['status'],'completed')
        self.assertEqual(len(list((target/'_incomplete').glob('*/ce/5shot/seed42/status.json'))),1)
        with patch('sys.argv',args), patch('utils.controlled_artifacts.load_splits',return_value=self.splits), \
                patch('run_controlled.run',side_effect=AssertionError('must skip')),contextlib.redirect_stdout(io.StringIO()):
            run_controlled.main()

    def test_runtime_mismatch_refuses_resume(self):
        import run_controlled
        path = self.runs/'protocol.json'
        original = path.read_text()
        try:
            identity = json.loads(original)
            identity['environment']['packages']['torch'] = 'different'
            path.write_text(json.dumps(identity))
            (self.root/'protocol.json').write_text(json.dumps(self.protocol))
            args = ['run_controlled','--protocol',str(self.root/'protocol.json'),'--model-path',str(self.snapshot),
                    '--output-dir',str(self.runs),'--device','cpu']
            with patch('sys.argv',args):
                with self.assertRaisesRegex(ValueError,'different protocol/model/runtime'): run_controlled.main()
        finally:
            path.write_text(original)


    def make_tied_run_copy(self, name):
        from analysis.audit_artifacts import frame_digest
        from utils.artifacts import RunArtifacts
        from utils.controlled_artifacts import file_hashes
        from utils.metrics import classification_metrics
        target = self.root/name
        shutil.copytree(self.runs,target)
        val = pd.concat([self.frame,self.frame.iloc[:2]],ignore_index=True)
        splits = (self.frame,val,val,self.categories)
        for arm in ARMS:
            directory = target/arm/'5shot/seed42'
            heads = torch.load(directory/'model/heads.pt',weights_only=True)
            heads['classifier']['weight'].zero_()
            heads['classifier']['bias'].zero_()
            torch.save(heads,directory/'model/heads.pt')
            (directory/'model/manifest.json').write_text(json.dumps(file_hashes(directory/'model')))
            config = read_json(directory/'config.json')
            config.update(validation_rows=len(val),test_rows=len(val),data_fingerprints={
                k:frame_digest(f) for k,f in [('train',self.frame),('validation',val),('test',val)]})
            (directory/'config.json').write_text(json.dumps(config))
            truth = np.array([self.categories.index(c) for c in val.category])
            old_ids = np.full(len(val),2)
            old_metrics = classification_metrics(truth,old_ids,self.categories)
            true_f1 = classification_metrics(truth,np.zeros(len(val),dtype=int),self.categories)['macro_f1']
            artifact = RunArtifacts(arm,'5shot',42,root=target)
            scores = np.full((len(val),3),1/3,dtype=np.float32)
            for filename in ('validation_predictions.csv','predictions.csv'):
                artifact.write_predictions(val.text.tolist(),truth,scores,self.categories,filename=filename)
                pred = pd.read_csv(directory/filename)
                pred['y_pred_id'],pred['y_pred'],pred['top2'],pred['top3'] = 2,self.categories[2],self.categories[1],self.categories[0]
                pred.to_csv(directory/filename,index=False)
            artifact.write_confusion(truth,old_ids,self.categories)
            metrics = read_json(directory/'metrics.json')
            metrics.update(validation=old_metrics,test=old_metrics,best_step=1,best_validation_macro_f1=true_f1)
            (directory/'metrics.json').write_text(json.dumps(metrics))
            history = read_json(directory/'training_history.json')
            for entry in history:
                entry['validation_macro_f1'] = true_f1
            (directory/'training_history.json').write_text(json.dumps(history))
        return target,splits

    def test_saved_checkpoint_recovery_repairs_ties_without_training(self):
        from analysis.repair_prediction_ranking import repair
        target,splits = self.make_tied_run_copy('tied-recovery')
        with patch('utils.controlled_artifacts.load_splits',return_value=splits):
            with self.assertRaisesRegex(ValueError,'Selected validation score mismatch'):
                validate_run(target/'ce/5shot/seed42')
        with patch('utils.controlled_artifacts.load_splits',return_value=splits), \
             patch('analysis.repair_prediction_ranking.load_splits',return_value=splits), \
             patch('supcon.joint.train_loop',side_effect=AssertionError('recovery must not train')), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(repair(target,seeds=[42],device='cpu'),3)
            self.assertEqual(repair(target,seeds=[42],device='cpu'),0)
            build_report(target,seeds=[42])
        for arm in ARMS:
            directory = target/arm/'5shot/seed42'
            self.assertTrue((pd.read_csv(directory/'predictions.csv').y_pred_id==0).all())
            self.assertEqual(len(list((directory/'ranking_backup').glob('*/metrics.json'))),1)
            self.assertEqual(read_json(directory/'metrics.json')['ranking_repair']['changes']['validation']['changed_predictions'],8)

    def test_recovery_refuses_unreproduced_history_without_changing_originals(self):
        from analysis.repair_prediction_ranking import repair
        target,splits = self.make_tied_run_copy('tied-abort')
        directory = target/'ce/5shot/seed42'
        history = read_json(directory/'training_history.json')
        for entry in history: entry['validation_macro_f1'] = .99
        (directory/'training_history.json').write_text(json.dumps(history))
        metrics = read_json(directory/'metrics.json')
        metrics['best_validation_macro_f1'] = .99
        (directory/'metrics.json').write_text(json.dumps(metrics))
        original = (directory/'predictions.csv').read_bytes()
        with patch('utils.controlled_artifacts.load_splits',return_value=splits), \
             patch('analysis.repair_prediction_ranking.load_splits',return_value=splits), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError,'Selected validation score mismatch'):
                repair(target,seeds=[42],device='cpu')
        self.assertEqual((directory/'predictions.csv').read_bytes(),original)
        self.assertFalse((directory/'ranking_backup').exists())
