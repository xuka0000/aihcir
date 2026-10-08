"""Separate TEST raw outputs from DEV-visible artifacts until legal release."""
import json
from pathlib import Path
from prepare_oct07_image_campaign import sha

def directory(root,kind,run_class):
    root=Path(root);c=json.loads((root/'contract.json').read_text())
    sealed=run_class=='formal' and c.get('test_seal',{}).get('required') is True
    if sealed and (root/'test_results_release.json').exists():
        from oct07_baseline_freeze import require_test_release
        require_test_release(root)
        sealed=False
    return root/('.sealed_test/'+kind if sealed else kind)

def prediction_path(root,row):return directory(root,'predictions',row['run_class'])/(row['task_id']+'.json')

def publish_released(root):
    from oct07_baseline_freeze import require_test_release
    root=Path(root);require_test_release(root)
    for kind in ('predictions','failures','calls','logs','native_bbox'):
        source=root/'.sealed_test'/kind
        if not source.exists():continue
        for path in (source.rglob('*') if kind=='native_bbox' else source.glob('*')):
            if path.is_dir() and kind=='native_bbox':continue
            if not path.is_file():raise ValueError('Unexpected nested sealed artifact')
            target=root/('pipeline_logs' if kind=='logs' else kind)/path.relative_to(source)
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                if sha(target)!=sha(path):raise ValueError('Released output identity mismatch')
                path.unlink()
            else:path.rename(target)
