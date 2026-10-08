"""Boundary tests: access, author control flow, accounting and immutable tasks."""
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from PIL import Image
from oct07_image_core import BudgetExhausted,EpisodeCalls,Packet,PublicEpisode,ViewAccess,WireLedger
from oct07_author_adapters import ADAPTERS,parse_output
from run_oct07_image_campaign import matrix,setup_db,claim
from prepare_oct07_image_campaign import OUT

class ImageCampaignTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.bundle=dict(episode_id='train_100000',instruction='Find the red car near and east of the yellow van.',cutoff_round=1,views=[],candidates=[])
        for t in (0,1,2):
            for a in (0,1):
                name=f'round{t:03d}_UAV_{a}.png';im=Image.new('RGB',(640,480),(100+t*20,50+a*30,100));im.save(self.root/name)
                import hashlib
                self.bundle['views'].append(dict(id=name,agent=f'UAV_{a}',round=t,sha256=hashlib.sha256((self.root/name).read_bytes()).hexdigest(),camera_world_position=[a*20,0,40]))
                self.bundle['candidates'].append(dict(id=f'old_{t}_{a}',role='target' if a else 'van',xy=[a*20,0],score=.7,span_m=4.,entity=dict(view_id=name,bbox_xyxy=[100,100,120,120]),sources=[name],detector_evidence=[]))
        self.ep=PublicEpisode(self.bundle,self.root);self.ledger=WireLedger(self.ep.agents,100000);self.access=ViewAccess(self.ep,self.ledger)
    def tearDown(self):self.temp.cleanup()
    def test_private_fields_rejected_and_future_removed(self):
        bad=dict(self.bundle,private_truth={})
        with self.assertRaises(ValueError):PublicEpisode(bad,self.root)
        self.assertTrue(all(v['round']<2 for v in self.ep.views.values()))
        with self.assertRaises(ValueError):self.access.advance(2)
    def test_sender_cannot_export_other_or_future(self):
        local=self.ep.local_candidates('UAV_1',0);cid=next(iter(local))
        with self.assertRaises(ValueError):Packet.proposal(self.ep,'UAV_0','UAV_1',cid,0)
        future=next(k for k,c in self.ep.candidates.items() if self.ep.views[c['entity']['view_id']]['round']==1)
        with self.assertRaises(ValueError):Packet.proposal(self.ep,'UAV_1','UAV_0',future,0)
    def test_wire_access_actual_crop_and_bytes(self):
        cid=next(iter(self.ep.local_candidates('UAV_1',0)));packet=Packet.proposal(self.ep,'UAV_1','UAV_0',cid,0)
        before=set(self.access.candidates());self.assertNotIn(cid,before)
        self.assertEqual(len(packet.wire),8+len(__import__('oct07_image_core').canonical(packet.header))+len(packet.payload))
        self.assertTrue(self.ledger.send(packet));self.assertIn(cid,self.access.candidates())
        names,images=self.access.images();self.assertIn((224,224),[im.size for im in images])
        for im in images:im.close()
        self.assertEqual(self.ledger.bytes,len(packet.wire))
    def test_rejected_and_dropped_packets_never_expose_pixels(self):
        cid=next(iter(self.ep.local_candidates('UAV_1',0)));packet=Packet.proposal(self.ep,'UAV_1','UAV_0',cid,0)
        for ledger in (WireLedger(self.ep.agents,len(packet.wire)-1),WireLedger(self.ep.agents,100000,drop_probability=1)):
            access=ViewAccess(self.ep,ledger);self.assertFalse(ledger.send(packet));self.assertNotIn(cid,access.candidates())
        self.assertEqual(ledger.bytes,len(packet.wire))
    def test_failed_sample_attempt_is_charged(self):
        class Bad:
            def generate(self,*a,**kw):raise RuntimeError('expected backend failure')
        calls=EpisodeCalls(Bad(),self.access,self.root/'calls.jsonl')
        with self.assertRaises(RuntimeError):calls('x',n=5)
        self.assertEqual(calls.used,5)
        with self.assertRaises(BudgetExhausted):calls('x',n=28)
        self.assertEqual(calls.used,5)
    def test_no_output_repair(self):
        with self.assertRaises(ValueError):parse_output(json.dumps(dict(assignment={'target':'unobserved','van':None})),self.access)
    def test_fixed_matrix_and_no_duplicate_claim(self):
        contract,rows=matrix(OUT);self.assertEqual(len(rows),39040)
        db=setup_db(self.root,rows[:2]);self.assertTrue(claim(db,rows[0]));self.assertFalse(claim(db,rows[0]));db.close()
    def test_independent_guard_single_mutation_and_readback(self):
        import deploy_oct07_image_campaign as deploy
        class API:
            def __init__(self):self.on=0;self.off=0;self.state='shutdown'
            def status(self):return self.state
            def power_on(self):self.on+=1;self.state='running'
            def power_off(self):self.off+=1;self.state='shutdown'
        api=API();clock=[0.]
        (self.root/'guard_context.json').write_text(json.dumps(dict(nonce='test',max_powered_seconds=7200)))
        (self.root/'owner.json').write_text(json.dumps(dict(nonce='test',desired='run',heartbeat=__import__('time').time())))
        def sleep(seconds):
            clock[0]+=seconds
            if clock[0]>=4:(self.root/'owner.json').write_text(json.dumps(dict(nonce='test',desired='stop',heartbeat=__import__('time').time())))
        with patch.object(deploy,'Provider',return_value=api),patch.object(deploy.time,'monotonic',side_effect=lambda:clock[0]),patch.object(deploy.time,'sleep',side_effect=sleep):deploy.guard(self.root)
        state=json.loads((self.root/'guard.json').read_text());self.assertEqual((api.on,api.off),(1,1));self.assertTrue(state['shutdown_confirmed'])
    def test_independent_guard_first_failure_stops_cloud_actions(self):
        import deploy_oct07_image_campaign as deploy
        class API:
            def status(self):raise OSError('expected first failure')
            def power_on(self):raise AssertionError('must never start')
            def power_off(self):raise AssertionError('must never mutate')
        (self.root/'guard_context.json').write_text(json.dumps(dict(nonce='test',max_powered_seconds=7200)))
        with patch.object(deploy,'Provider',return_value=API()):deploy.guard(self.root)
        state=json.loads((self.root/'guard.json').read_text());self.assertTrue(state['cloud_checks_stopped']);self.assertEqual(state['power_on_attempts'],0)
    def test_guard_recovers_local_sharing_collision(self):
        import deploy_oct07_image_campaign as deploy
        with patch.object(deploy.Path,'read_text',side_effect=[PermissionError('sharing collision'),'{"nonce":"same"}']),patch.object(deploy.time,'sleep'):
            self.assertEqual(deploy.read_guard_json(self.root/'owner.json')['nonce'],'same')
        with patch.object(deploy.Path,'read_text',side_effect=PermissionError('persistent denial')),patch.object(deploy.time,'sleep'):
            with self.assertRaises(PermissionError):deploy.read_guard_json(self.root/'owner.json')
    def test_existing_guard_never_starts_and_keeps_absolute_deadline(self):
        import deploy_oct07_image_campaign as deploy
        class API:
            def __init__(self):self.on=0;self.off=0;self.state='running'
            def status(self):return self.state
            def power_on(self):self.on+=1;raise AssertionError('Existing task must never be started')
            def power_off(self):self.off+=1;self.state='shutdown'
        api=API();clock=[0.]
        (self.root/'guard_context.json').write_text(json.dumps(dict(nonce='same',max_powered_seconds=7200,attach_existing=True,effective_deadline=184,existing_job_identity=dict(pid=884,start_ticks='original'))))
        (self.root/'owner.json').write_text(json.dumps(dict(nonce='same',desired='run',heartbeat=0)))
        def sleep(seconds):clock[0]+=seconds
        with patch.object(deploy,'Provider',return_value=api),patch.object(deploy.time,'time',side_effect=lambda:clock[0]),patch.object(deploy.time,'monotonic',side_effect=lambda:clock[0]),patch.object(deploy.time,'sleep',side_effect=sleep):deploy.guard(self.root)
        state=json.loads((self.root/'guard.json').read_text());self.assertEqual((api.on,api.off),(0,1));self.assertTrue(state['shutdown_confirmed']);self.assertLess(clock[0],90)
    def test_external_budget_guard_arms_without_start_and_stops_external_start(self):
        import deploy_oct07_image_campaign as deploy
        class API:
            def __init__(self):self.on=0;self.off=0;self.state='shutdown'
            def status(self):return self.state
            def power_on(self):self.on+=1;raise AssertionError('Only shared CLI may start')
            def power_off(self):self.off+=1;self.state='shutdown'
        api=API();clock=[0.]
        (self.root/'guard_context.json').write_text(json.dumps(dict(nonce='budget',max_powered_seconds=1200,external_budget_start=True,effective_deadline=1200)))
        # An external start belongs to this guard only with a task-bound lease.
        # Without it the production guard correctly refuses to stop another job.
        (self.root/'budget_lease.json').write_text(json.dumps(dict(lease_id='test-only-lease')))
        (self.root/'owner.json').write_text(json.dumps(dict(nonce='budget',desired='run',heartbeat=0)))
        def sleep(seconds):
            clock[0]+=seconds
            if clock[0]==4:api.state='running'
            if clock[0]>=8:(self.root/'owner.json').write_text(json.dumps(dict(nonce='budget',desired='stop',heartbeat=clock[0])))
        def owned_stop(folder,context,state,save):
            self.assertTrue((folder/'budget_lease.json').is_file())
            api.power_off()
            save(phase='shutdown_confirmed',terminal=True,shutdown_confirmed=api.status()=='shutdown')
        with patch.object(deploy,'Provider',return_value=api),patch.object(deploy,'shutdown_once_and_verify',side_effect=owned_stop) as stop,patch.object(deploy.time,'time',side_effect=lambda:clock[0]),patch.object(deploy.time,'monotonic',side_effect=lambda:clock[0]),patch.object(deploy.time,'sleep',side_effect=sleep):deploy.guard(self.root)
        stop.assert_called_once()
        result=json.loads((self.root/'guard.json').read_text())
        self.assertEqual((api.on,api.off),(0,1));self.assertTrue(result['shutdown_confirmed'])
    def test_external_guard_cancelled_before_paid_start_never_mutates(self):
        import deploy_oct07_image_campaign as deploy
        class API:
            def status(self):return 'shutdown'
            def power_on(self):raise AssertionError('No paid start')
            def power_off(self):raise AssertionError('Already shutdown')
        (self.root/'guard_context.json').write_text(json.dumps(dict(nonce='budget',max_powered_seconds=1200,external_budget_start=True,effective_deadline=__import__('time').time()+1200)))
        (self.root/'owner.json').write_text(json.dumps(dict(nonce='budget',desired='stop',heartbeat=__import__('time').time())))
        with patch.object(deploy,'Provider',return_value=API()):deploy.guard(self.root)
        self.assertTrue(json.loads((self.root/'guard.json').read_text())['shutdown_confirmed'])
    def test_all_author_controllers_execute_their_sources(self):
        class Fake:
            key='test_backend'
            def __init__(self,access):self.access=access;self.used=0;self.outputs=[]
            def __call__(self,prompt,n=1,**kw):
                self.used+=n
                if self.used>32:raise BudgetExhausted()
                answer=json.dumps(dict(assignment={'target':None,'van':None},observed_direction=None))
                if 'the best choice is N' in prompt:out='the best choice is 1'
                elif '# There is an error in the code' in prompt:out='It is correct.\ndef solution():\n    return '+repr(dict(assignment={'target':None,'van':None},observed_direction=None))
                elif '# solution using Python:' in prompt:out='def solution():\n    return '+repr(dict(assignment={'target':None,'van':None},observed_direction=None))
                elif 'Recent actions:' in prompt:out='Finalize current bindings'
                elif 'Send a concise helpful message' in prompt:out='"observed evidence remains uncertain"'
                elif 'Return {"reason":"...","plan"' in prompt:out='{"reason":"observe","plan":["ground"]}'
                elif 'actor JSON schema' in prompt:out=json.dumps({"UAV_0's action":'Finish['+answer+']',"UAV_1's action":'Done','reason':'submit','subtask':'ground','memory':'observed','failure reason':''})
                elif 'completed subtasks' in prompt:out='{"reason":"no hidden check","completed subtasks":[]}'
                elif kw.get('assistant_prefix')=='Thought 1:':out='Submit observed roles.\nAction 1: Finish['+answer+']'
                elif 'Continue this selected plan' in prompt:out='Answer:\n'+answer
                elif 'Plan:\nGive a plan' in prompt:out='Check observations.\nAnswer:\n'
                else:out=answer
                self.outputs.extend([out]*n);return [out]*n
        inherited=json.loads((OUT/'contract.json').read_text(encoding='utf-8'))['sources']['author_methods']
        for method in inherited:
            fn=ADAPTERS[method]
            with self.subTest(method=method):
                access=ViewAccess(self.ep,WireLedger(self.ep.agents,100000))
                calls=Fake(access);calls.vlm=SimpleNamespace(key='test_backend');calls.records=[]
                result,trace=fn(OUT,calls)
                self.assertTrue(result['valid']);self.assertTrue(trace['author_functions']);self.assertGreater(calls.used,0)
                if method=='tot':self.assertEqual(calls.used,20)

if __name__=='__main__':unittest.main(verbosity=2)
