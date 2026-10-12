import unittest
import pandas as pd
from analysis.audit_artifacts import check_confusion
from utils.metrics import classification_metrics

class ArtifactAuditTest(unittest.TestCase):
    def test_detects_corrupted_metric_and_counts(self):
        labels=['a','b']
        matrix=pd.DataFrame([[39,1],[2,38]],index=labels,columns=labels)
        metrics=classification_metrics([0]*40+[1]*40,[0]*39+[1]+[0]*2+[1]*38,labels)
        self.assertEqual(check_confusion(matrix,metrics,labels),[])
        metrics['macro_f1']=1.0
        self.assertIn('test macro_f1 mismatch',check_confusion(matrix,metrics,labels))
        matrix.iloc[0,0]=-1
        self.assertEqual(check_confusion(matrix,metrics,labels),['invalid confusion counts'])

    def test_rejects_reordered_labels(self):
        matrix=pd.DataFrame([[40,0],[0,40]],index=['b','a'],columns=['b','a'])
        self.assertEqual(check_confusion(matrix,{},['a','b']),['confusion label order mismatch'])
