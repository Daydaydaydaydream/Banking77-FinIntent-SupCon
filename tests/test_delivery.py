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
