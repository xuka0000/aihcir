"""Freeze a DEV-reviewed scientific version without reading TEST results."""
import json
from pathlib import Path
import sqlite3
import time
from prepare_oct07_image_campaign import sha,write
from run_oct07_image_campaign import matrix,require_selected_version
from oct07_runtime_scope import queue_path,enforce_prior_first_freeze

def select(root,code=None):
    root=Path(root);code=Path(code or Path(__file__).parent);c,rows=matrix(root)
    enforce_prior_first_freeze(root,code)
    if (root/'test_results_release.json').exists():raise ValueError('TEST already opened; no new scientific selection')
    targets=json.loads((root/'baseline_targets.json').read_text())
    if targets.get('baseline_rows_reviewed')!=c['units']['public_baseline_development'] or targets.get('selection_split')!='development':
        raise ValueError('Complete public DEV targets are required')
    from oct07_fit_identity import verify_fits
    fits=verify_fits(root)
    if len(fits)!=c['units']['training_fits']:raise ValueError('Complete frozen fit identity required')
    db=sqlite3.connect('file:'+str(queue_path(root).resolve())+'?mode=ro',uri=True)
    try:
        states=dict(db.execute('SELECT id,state FROM jobs'))
        dev=[r for r in rows if r['run_class']=='development']
        if len(dev)!=c['units']['development_rows'] or any(states.get(r['task_id']) not in ('completed','failed') for r in dev):
            raise ValueError('Full DEV matrix must be terminal before selection')
        review=json.loads((root/'development_mechanism_review.json').read_text())
        if review.get('selection_split')!='development' or review.get('rows_reviewed')!=len(dev) or review.get('test_opened') is not False:
            raise ValueError('Actual full DEV mechanism review is required; do not auto-approve from counts')
        from oct07_baseline_freeze import baseline_files
        identity=baseline_files(code)
        for name in ('train_oct07_image_selector.py','oct07_trace_selector.py','oct07_fit_identity.py','oct07_stage_freeze.py'):
            identity[name]=sha(code/name)
        value=dict(frozen=True,selection_split='development',development_rows_reviewed=len(dev),
            implementation_sha256=identity,matrix_sha256=sha(root/'matrix.csv'),fit_identity=fits,
            baseline_targets_sha256=sha(root/'baseline_targets.json'),mechanism_review_sha256=sha(root/'development_mechanism_review.json'),
            seeds=c['training']['seeds'],checkpoint_selection='final_epoch',selected_method='trace_full',
            test_results_opened=False,observed_at=time.time(),scientific_acceptance=False)
        p=root/'algorithm_selection.json'
        if p.exists():
            require_selected_version(root,c,db)
            old=json.loads(p.read_text())
            for k in ('matrix_sha256','implementation_sha256','baseline_targets_sha256','mechanism_review_sha256','fit_identity'):
                if old[k]!=value[k]:raise ValueError('Existing DEV selection differs; no overwrite')
            return old
        write(p,value,True)
        require_selected_version(root,c,db)
        return value
    finally:db.close()
