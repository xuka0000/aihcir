"""Sole power-action owner for one existing task-owned AutoDL node.

The guard is a deterministic helper, not an AI worker. It needs the local
workstation and network to remain available; it cannot guarantee offline power
control. No token or platform snapshot is serialized in its artifacts.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import time
import ssl
from requests.adapters import HTTPAdapter

import requests

MAX_POWERED_SECONDS=2400
STOP_RESERVE_SECONDS=120
OWNER_STALE_SECONDS=90
STOPPED_STATUSES=frozenset(('shutdown','shutdown_by_starting_error','shutdown_by_running_error'))


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    # A concurrent Windows reader can briefly prevent atomic replacement.
    for attempt in range(4):
        try:temporary.replace(path);return
        except PermissionError as exc:
            if getattr(exc,'winerror',None)!=5 or attempt==3:raise
            time.sleep(.01)


class ProviderTLS12(HTTPAdapter):
    def init_poolmanager(self,*args,**kwargs):
        context=ssl.create_default_context()
        context.maximum_version=ssl.TLSVersion.TLSv1_2
        kwargs['ssl_context']=context
        return super().init_poolmanager(*args,**kwargs)


class ProviderAPIError(RuntimeError):
    """Allowlisted, redacted provider receipt; never retain a response/request."""
    def __init__(self, endpoint, method, status, value, token):
        def clean(text, limit):
            if not isinstance(text,(str,int,float)):return None
            text=str(text)
            if token:text=text.replace(token,'[REDACTED]')
            text=re.sub(r'(?i)\bBearer\s+[^\s,;]+','Bearer [REDACTED]',text)
            text=re.sub(r'''(?ix)(["']?(?:authorization|(?:root|ssh)[-_]?password|password|(?:access|refresh|jupyter|auth)[-_]?token|token|(?:x[-_])?api[-_]?key|secret)["']?\s*[:=]\s*)(?:"[^"]*"|'[^']*'|[^\s,;]+)''',r'\1[REDACTED]',text)
            return ''.join(c if c.isprintable() else ' ' for c in text)[:limit]
        value=value if isinstance(value,dict) else {}
        self.receipt=dict(endpoint=endpoint,method=method,http_status=status,
                          code=clean(value.get('code'),128),
                          message=clean(value.get('msg'),512),
                          request_id=clean(value.get('request_id'),128))
        super().__init__('AutoDL request failed: '+json.dumps(self.receipt,ensure_ascii=False))


class OwnedControl:
    def __init__(self, script):
        spec=importlib.util.spec_from_file_location('owned_autodl_control',script)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        rows=[r for r in module.registry() if r.get('role')=='sota_run' and r.get('gpu_spec')=='5090-p']
        if len(rows)!=1:raise ValueError('Exactly one task-owned model node is required')
        self.uid,self.token_path=rows[0]['uuid'],module.TOKEN
        self.node_hash=hashlib.sha256(self.uid.encode()).hexdigest()
        # Verified direct TLS1.2 avoids reproducible TLS1.3 record errors.
        # Certificate validation stays enabled. SSH keeps its required proxy.
        self.session=requests.Session()
        route=getattr(module,'API_ROUTE','direct')
        if route not in ('direct','configured_proxy'):raise ValueError('Unknown provider API route')
        self.session.trust_env=route=='configured_proxy'
        self.api_route=route
        self.session.mount('https://',ProviderTLS12())
        self.transport_read_retries=0

    def api(self, endpoint, method='GET', **body):
        # Same credential, no fallback/discovery. Only read-only GET transport
        # failures get two bounded retries. Mutations are never replayed.
        token=self.token_path.read_text(encoding='utf-8').strip()
        url='https://api.autodl.com/api/v1/dev/'+endpoint
        for attempt in range(3 if method=='GET' else 1):
            try:
                response=self.session.request(method,url,headers={'Authorization':token},timeout=(3,8),
                    **({'params':body} if method=='GET' else {'json':body}))
                try:value=response.json()
                except ValueError:value=None
                if (not 200<=response.status_code<300 or not isinstance(value,dict)
                        or value.get('code')!='Success'):
                    raise ProviderAPIError(endpoint,method,response.status_code,value,token)
                break
            except (requests.exceptions.ConnectionError,requests.exceptions.Timeout):
                if method!='GET' or attempt==2:raise
                self.transport_read_retries+=1;time.sleep(1)
        return value.get('data')

    def status(self):return self.api('instance/pro/status',instance_uuid=self.uid)
    def power_on(self):return self.api('instance/pro/power_on','POST',instance_uuid=self.uid,payload='gpu',start_command='sleep 1')
    def power_off(self):return self.api('instance/pro/power_off','POST',instance_uuid=self.uid)
    def connection(self):return self.api('instance/pro/snapshot',instance_uuid=self.uid)


class PowerGuard:
    def __init__(self, api, nonce, *, clock=time.monotonic, wall=time.time, checkpoint=lambda _:None,
                 max_powered_seconds=MAX_POWERED_SECONDS, ceiling_seconds=MAX_POWERED_SECONDS):
        if ceiling_seconds not in (MAX_POWERED_SECONDS,7200):
            raise ValueError('Unknown campaign ceiling')
        if type(max_powered_seconds) is not int or not 120<=max_powered_seconds<=ceiling_seconds:
            raise ValueError('Power deadline must stay within the original ceiling')
        self.api,self.nonce,self.clock,self.wall,self.checkpoint=api,nonce,clock,wall,checkpoint
        self.max_powered_seconds=max_powered_seconds
        self.started=clock()
        self.state=dict(schema='public_mcq_power_guard_v1',nonce=nonce,state='initializing',
                        started_unix=wall(),power_on_intent=False,power_on_attempts=0,power_off_attempts=0,
                        shutdown_confirmed=False,terminal=False,max_powered_seconds=max_powered_seconds,
                        shutdown_begin_by_seconds=max_powered_seconds-STOP_RESERVE_SECONDS,automatic_model_retries=0,
                        unattended_supervision_available=False)
        self.stop_reason=None;self.last_off=-1e9
        self.shutdown_first_seen=None
        self.stop_deadline_seconds=max_powered_seconds

    def save(self):
        self.state.update(heartbeat_unix=self.wall(),elapsed_seconds=self.clock()-self.started)
        self.state['transport_read_retries']=getattr(self.api,'transport_read_retries',0)
        try:self.checkpoint(dict(self.state));return True
        except OSError as exc:
            self.state['logging_error_type']=type(exc).__name__
            self.stop_reason='guard_status_write_error'
            return False

    def record_error(self, key, exc):
        self.state[key+'_type']=type(exc).__name__
        if isinstance(exc,ProviderAPIError):
            # First error for each phase survives subsequent cleanup failures.
            self.state.setdefault(key+'_receipt',dict(exc.receipt))

    def arm(self):
        status=self.api.status()
        if status not in STOPPED_STATUSES:raise RuntimeError('Existing node is occupied or status unknown')
        self.state.update(state='armed',node_status=status)
        if not self.save():raise RuntimeError('Guard cannot publish readiness')

    def stop(self, reason):
        self.stop_reason=self.stop_reason or reason
        if not self.state['power_on_intent'] or self.state['power_on_attempts']==0:
            self.state.update(state='cancelled_before_power',stop_reason=self.stop_reason,terminal=True)
            self.save();return
        self.state.update(state='stopping',stop_reason=self.stop_reason);self.save()
        status=None
        try:status=self.api.status()
        except Exception as exc:self.record_error('stop_status_error',exc)
        # Always issue a stop after a possibly delivered power-on, even if the
        # first status still says shutdown. It can lag a queued boot request.
        if status in STOPPED_STATUSES and self.state['power_off_attempts']>0:
            if self.shutdown_first_seen is None:self.shutdown_first_seen=self.clock()
            if self.clock()-self.shutdown_first_seen>=10:
                self.state.update(state='shutdown_confirmed',node_status=status,shutdown_confirmed=True,terminal=True,
                                  shutdown_confirmed_unix=self.wall(),confirmation_scope='Two provider stopped-state observations at least10s apart, after a stop attempt')
        else:
            self.shutdown_first_seen=None
        need_off=(self.state['power_off_attempts']==0 or status not in STOPPED_STATUSES)
        if need_off and self.state['power_off_attempts']<3 and self.clock()-self.last_off>=15:
            try:
                self.state['power_off_attempts']+=1;self.last_off=self.clock();self.save()
                self.api.power_off()
            except Exception as exc:self.record_error('stop_error',exc)
        if not self.state['shutdown_confirmed'] and self.clock()-self.started>=self.stop_deadline_seconds:
            self.state.update(state='shutdown_unconfirmed',terminal=True)
        self.save()
    def tick(self, owner):
        if self.state['terminal']:return
        elapsed=self.clock()-self.started
        invalid=(not isinstance(owner,dict) or owner.get('nonce')!=self.nonce)
        if not invalid:
            heartbeat=owner.get('heartbeat_unix')
            invalid=(type(heartbeat) not in (int,float) or not 0<=self.wall()-heartbeat<=OWNER_STALE_SECONDS)
        reason=('owner_missing_or_stale' if invalid else
                'powered_deadline' if elapsed>=self.max_powered_seconds-STOP_RESERVE_SECONDS else
                'owner_stop' if owner.get('desired')=='stop' else self.stop_reason)
        if reason:
            self.stop(reason);return
        if owner.get('desired') not in ('idle','run'):
            self.stop('invalid_owner_request');return
        try:
            if owner['desired']=='run' and not self.state['power_on_intent']:
                self.state.update(state='power_on_requested',power_on_intent=True)
                if not self.save():raise RuntimeError('Power intent could not be persisted')
                self.state['power_on_attempts']=1
                self.api.power_on()  # Intent persists before any possible lost response.
            if self.state['power_on_attempts']:
                self.state['node_status']=self.api.status()
                if self.state['node_status'] in STOPPED_STATUSES-{'shutdown'}:
                    self.stop('provider_terminal_error');return
                self.state['state']='running' if self.state['node_status']=='running' else 'starting'
            self.save()
        except Exception as exc:
            self.record_error('error',exc)
            self.stop('power_or_status_error')


def emergency_shutdown(api, nonce, *, clock=time.monotonic, wall=time.time, sleep=time.sleep, checkpoint=lambda _:None):
    """Bounded cleanup after the original guard was reaped and lease verified.

    This does not reset the research clock or authorize any workload retry.
    """
    guard=PowerGuard(api,nonce,clock=clock,wall=wall,checkpoint=checkpoint)
    guard.stop_deadline_seconds=120
    guard.state.update(power_on_intent=True,power_on_attempts=1,emergency_cleanup=True,
                       max_cleanup_seconds=120,scope='cleanup_only_original_research_deadline_unchanged')
    while not guard.state['terminal']:
        guard.stop('guard_reaped_emergency_cleanup')
        if not guard.state['terminal']:sleep(2)
    return guard.state


def run_guard(folder,control_script):
    context=read(folder/'guard_context.json');api=OwnedControl(control_script)
    lease=Path(context['lease'])
    if api.node_hash!=context['node_hash'] or read(lease)['nonce']!=context['nonce']:
        raise ValueError('Guard ownership lease mismatch')
    def save(value):
        value['pid']=os.getpid();write(folder/'guard.json',value)
    ceiling=MAX_POWERED_SECONDS
    if context.get('max_powered_seconds',MAX_POWERED_SECONDS)>ceiling:
        contract=read(folder/'batch_contract.json')
        if (contract.get('schema')!='aircop_complete_dev_baseline_v1'
                or contract.get('powered_seconds')!=7200 or contract.get('model_calls')!=231
                or len(set(contract.get('episode_ids',[])))!=231
                or contract.get('automatic_model_retries')!=0):
            raise ValueError('Complete development campaign contract absent')
        ceiling=7200
    guard=PowerGuard(api,context['nonce'],checkpoint=save,
                     max_powered_seconds=context.get('max_powered_seconds',MAX_POWERED_SECONDS),ceiling_seconds=ceiling)
    try:
        guard.arm()
        while not guard.state['terminal']:
            try:owner=read(folder/'owner.json')
            except Exception:owner=None
            guard.tick(owner);time.sleep(2)
    except BaseException as exc:
        guard.record_error('error',exc)
        # Failures before any power intent cannot stop another workload.
        if guard.state['power_on_intent']:
            while not guard.state['terminal']:
                guard.stop('guard_exception');time.sleep(2)
        else:
            guard.state.update(state='guard_failed_before_power',terminal=True);guard.save()
    finally:
        if (guard.state['shutdown_confirmed'] or not guard.state['power_on_intent']) and lease.is_file() and read(lease)['nonce']==context['nonce']:
            lease.unlink()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',type=Path,required=True);parser.add_argument('--control-script',type=Path,required=True)
    args=parser.parse_args();run_guard(args.folder,args.control_script)
