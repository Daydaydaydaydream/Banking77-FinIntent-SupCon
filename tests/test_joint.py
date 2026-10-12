import json
import unittest
from pathlib import Path
import torch
from supcon.joint import joint_loss, train_loop, validate_protocol
from supcon.losses import hard_negative_weight_matrix, supervised_contrastive_loss

class JointTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.features=torch.randn(8,5,requires_grad=True)
        self.labels=torch.tensor([0,0,1,1,2,2,3,3])
        self.logits=torch.randn(8,4,requires_grad=True)

    def test_zero_weight_exactly_recovers_ce_and_gradients(self):
        plain=joint_loss(self.logits,self.features,self.labels,'ce')[0]
        joint=joint_loss(self.logits,self.features,self.labels,'ce_supcon_hn',weight=0)[0]
        self.assertTrue(torch.equal(plain,joint))
        self.assertTrue(torch.equal(torch.autograd.grad(plain,self.logits)[0],torch.autograd.grad(joint,self.logits)[0]))

    def test_hn_changes_loss_and_gradient(self):
        plain=joint_loss(self.logits,self.features,self.labels,'ce_supcon')[0]
        hard=joint_loss(self.logits,self.features,self.labels,'ce_supcon_hn')[0]
        self.assertFalse(torch.isclose(plain,hard))
        self.assertFalse(torch.allclose(torch.autograd.grad(plain,self.features)[0],torch.autograd.grad(hard,self.features)[0]))

    def test_uniform_weights_recover_supcon_and_cold_weights_are_finite(self):
        plain=supervised_contrastive_loss(self.features,self.labels)
        uniform=supervised_contrastive_loss(self.features,self.labels,hard_negative_weights=torch.ones(8,8))
        self.assertTrue(torch.allclose(plain,uniform))
        features=torch.nn.functional.normalize(self.features,dim=1)
        weights=hard_negative_weight_matrix(features@features.T,self.labels,0.001)
        loss=supervised_contrastive_loss(features,self.labels,0.001,weights)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(self.features.grad).all())
        self.assertFalse(weights.requires_grad)

    def test_fixed_updates_and_best_complete_state_restored(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder=torch.nn.Linear(2,5)
                self.classifier=torch.nn.Linear(5,4)
            def forward(self,input_ids,attention_mask):
                hidden=self.encoder(input_ids.float())
                return hidden,self.classifier(hidden)
        model=Tiny()
        batch=dict(input_ids=torch.randn(8,2),attention_mask=torch.ones(8,2),labels=self.labels)
        states=[]
        def evaluate(model):
            states.append({k:v.detach().clone() for k,v in model.state_dict().items()})
            return [0.9,0.1,0.2][len(states)-1]
        protocol=dict(max_steps=3,eval_steps=1,supcon_weight=.1,supcon_temperature=.1,hn_temperature=.5)
        history,extra=train_loop(model,torch.optim.AdamW(model.parameters(),lr=.01),lambda epoch:[batch],evaluate,arm='ce_supcon_hn',protocol=protocol,device='cpu',resolved_precision='fp32')
        self.assertEqual(extra['actual_steps'],3)
        self.assertEqual(extra['best_step'],1)
        self.assertEqual(len(history),3)
        for k,v in model.state_dict().items(): self.assertTrue(torch.equal(v,states[0][k]))
        self.assertFalse(torch.equal(states[0]['encoder.weight'],states[-1]['encoder.weight']))

    def test_protocol_rejects_zero_budget(self):
        protocol=json.loads((Path(__file__).resolve().parents[1]/'configs/controlled_v1.json').read_text())
        validate_protocol(protocol)
        protocol['max_steps']=0
        with self.assertRaises(ValueError): validate_protocol(protocol)


class JointArtifactIntegrationTest(unittest.TestCase):
    def test_real_tiny_encoder_artifacts_and_reload(self):
        import tempfile
        from unittest.mock import patch
        import pandas as pd
        from transformers import BertConfig, BertModel, BertTokenizerFast
        from supcon.joint import run, model_fingerprints, load_saved_model
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            snapshot=root/'snapshot'
            snapshot.mkdir()
            (snapshot/'vocab.txt').write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nrefund\ncard\nwhere\n')
            tokenizer=BertTokenizerFast(vocab_file=str(snapshot/'vocab.txt'))
            tokenizer.save_pretrained(snapshot)
            BertModel(BertConfig(vocab_size=8,hidden_size=8,num_hidden_layers=1,num_attention_heads=2,intermediate_size=16)).save_pretrained(snapshot)
            frame=pd.DataFrame({'text':['refund','where refund','card','where card','where','where where'],'category':['refund','refund','card','card','where','where']})
            protocol=json.loads((Path(__file__).resolve().parents[1]/'configs/controlled_v1.json').read_text())
            protocol.update(max_steps=2,eval_steps=1,projection_dim=4,eval_batch_size=4)
            hashes=model_fingerprints(snapshot)
            with patch('supcon.joint.load_splits',return_value=(frame,frame,frame,['refund','card','where'])):
                run('ce_supcon_hn','5shot',42,protocol=protocol,model_path=snapshot,model_hashes=hashes,root=root/'runs',device='cpu')
            directory=root/'runs/ce_supcon_hn/5shot/seed42'
            model,categories=load_saved_model(directory)
            inputs=tokenizer(frame.text.tolist(),padding=True,return_tensors='pt')
            with torch.no_grad():
                logits=model.classify(model.embed(inputs['input_ids'],inputs['attention_mask']))
            scores=torch.softmax(logits,dim=1).numpy()
            saved=pd.read_csv(directory/'predictions.csv')
            import numpy as np
            self.assertTrue(np.allclose(scores.max(axis=1),saved.top1_score,atol=1e-6))
            self.assertEqual(scores.argmax(axis=1).tolist(),saved.y_pred_id.tolist())
            self.assertTrue((directory/'validation_predictions.csv').exists())
            self.assertEqual(json.loads((directory/'status.json').read_text())['status'],'completed')
