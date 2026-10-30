import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1]
FAKE = '''import argparse,json,os
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--out'); p.add_argument('--last')
a,_=p.parse_known_args()
if os.environ.get('FAKE_ERROR'):
    raise OSError('Cannot read image file: /dataset/broken.png')
Path(a.out).write_bytes(b'fake-checkpoint')
Path(a.last).write_bytes(b'fake-last')
score=float(os.environ.get('FAKE_SCORE','0.91'))
print('METRIC_JSON '+json.dumps({'epoch':1,'val_f1':score,'metric':'validation_global_pixel_f1_training_resolution'}),flush=True)
'''
class SupervisorTests(unittest.TestCase):
    def invoke(self, score='0.91', error=False, max_runs=1):
        import os
        temp=tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        root=Path(temp.name)
        shutil.copy2(SOURCE/'train_until_target.py',root/'train_until_target.py')
        (root/'train_plaque_unet.py').write_text(FAKE)
        (root/'data/train').mkdir(parents=True); (root/'data/val').mkdir()
        env=dict(os.environ,FAKE_SCORE=score)
        if error: env['FAKE_ERROR']='1'
        r=subprocess.run([sys.executable,str(root/'train_until_target.py'),'--data',str(root/'data'),
            '--output',str(root/'result'),'--max-runs',str(max_runs)],capture_output=True,text=True,env=env,timeout=15)
        s=json.loads((root/'result/status.json').read_text())
        events=[json.loads(x) for x in (root/'result/events.jsonl').read_text().splitlines()]
        return r,s,events,root
    def test_target(self):
        r,s,e,_=self.invoke(); self.assertEqual(r.returncode,0,r.stderr)
        self.assertTrue(s['target_reached']); self.assertTrue(Path(s['best_checkpoint']).is_file())
        self.assertTrue(all(x['time'].endswith('+07:00') for x in e))
    def test_unrounded(self):
        r,s,_,_=self.invoke('0.899999'); self.assertEqual(r.returncode,2)
        self.assertFalse(s['target_reached'])
    def test_file_error(self):
        r,s,_,root=self.invoke(error=True); self.assertEqual(r.returncode,1)
        self.assertIn('/dataset/broken.png',s['error']['data_files'])
        self.assertEqual(s['error']['source_file'],str(root/'train_plaque_unet.py'))
        self.assertTrue(s['error']['time'].endswith('+07:00'))
    def test_nan(self):
        r,s,_,_=self.invoke('nan'); self.assertEqual(r.returncode,1)
        self.assertEqual(s['status'],'error')
    def test_multiple_trials(self):
        r,s,e,_=self.invoke('0.7',max_runs=2); self.assertEqual(r.returncode,2)
        self.assertEqual(s['trial'],2)
        starts=[x for x in e if x['event']=='TRIAL_START'];self.assertEqual(len(starts),2)
        self.assertNotEqual(starts[0]['model_file'],starts[1]['model_file'])
if __name__=='__main__': unittest.main()
