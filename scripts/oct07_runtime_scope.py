"""Prospectively declared first reporting view, with one parent task ledger."""
import copy
import csv
from collections import Counter
import json
from pathlib import Path
from prepare_oct07_image_campaign import sha


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def scoped_contract(parent, plan, matrix_sha, spec_hashes):
    value = copy.deepcopy(parent)
    value['campaign_id'] = plan['scope_id']
    value['matrix_sha256'] = matrix_sha
    value['models'] = {'qwen3': value['models']['qwen3']}
    value['data'].update(formal_specs=48, formal_per_stratum=2, capture_scenes=192)
    value['units'].update(image_episode_rows=7560, development_rows=2520, formal_rows=5040,
        training_fits=12, native_bbox_frame_rows=864, public_baseline_development=1656,
        public_baseline_formal=3312)
    value['training'].update(fits=12)
    # This is the declared reporting view, not the paused parent matrix.
    # Counts describe existing rows only; they do not add sensitivity runs.
    value['counts'] = {'development/main': 2040, 'development/ablation': 432,
        'development/all_view_reference': 48, 'formal/main': 4080,
        'formal/ablation': 864, 'formal/all_view_reference': 96}
    value['baseline_first'].update(public_rows=1656, baseline_stage_rows=1800)
    value['specification_sha256'] = dict(spec_hashes)
    value['reporting_scope'] = dict(id=plan['scope_id'], parent_directory='..',
        parent_contract_sha256=plan['parent']['contract_sha256'],
        parent_matrix_sha256=plan['parent']['matrix_sha256'],
        task_ledger='parent/queue.sqlite', parent_fit_ceiling=24,
        selection_basis='First two frozen TEST IDs per24 metadata strata; no outcomes',
        first_TEST_requires_complete_DEV=2520, first_TEST_requires_complete_formal=5040,
        parent_TEST_gate_unchanged=True, later_methods_must_match_first_DEV_freeze=True)
    return value


def repair_derived_metadata(root):
    """Correct inherited display counts only, before any scientific freeze."""
    root = Path(root).resolve()
    c = read(root/'contract.json')
    parent = parent_root(root)
    plan = read(parent/'first_manuscript_protocol.json')
    specs = {s: sha(root/'specifications'/(s+'.specs.jsonl'))
             for s in ('train', 'validation', 'test')}
    expected = scoped_contract(read(parent/'contract.json'), plan,
                               plan['matrix_sha256'], specs)
    if c == expected:
        return verify_runtime(root)
    for name in ('baseline_freeze.json', 'algorithm_selection.json',
                 'test_results_release.json'):
        if (root/name).exists():
            raise ValueError('Derived metadata is already scientifically frozen: '+name)
    # Reject every other difference, including even one changed sample budget.
    comparison = copy.deepcopy(c)
    comparison['counts'] = expected['counts']
    comparison['baseline_first'].update(public_rows=1656, baseline_stage_rows=1800)
    if comparison != expected or sha(root/'matrix.csv') != plan['matrix_sha256']:
        raise ValueError('Only inherited reporting counts may be corrected')
    before = sha(root/'contract.json')
    from prepare_oct07_image_campaign import write
    import time
    receipt = dict(observed_at=time.time(), before_sha256=before,
        previous_counts=c['counts'], previous_baseline_first=c['baseline_first'],
        matrix_sha256=plan['matrix_sha256'], parent_contract_sha256=sha(parent/'contract.json'),
        scope='Derived counts only; matrix, seeds, budgets, parent pause and all192-scene freeze unchanged',
        scientific_results_created=0)
    # Atomic replacement avoids changing any hard-linked historical source.
    temporary = root/'contract.derived-counts.tmp'
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(expected, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    temporary.replace(root/'contract.json')
    receipt['after_sha256'] = sha(root/'contract.json')
    write(root/'derived_metadata_repair.json', receipt)
    return verify_runtime(root)


def parent_root(root):
    root = Path(root).resolve()
    if not (root/'contract.json').is_file():
        return root
    c = read(root/'contract.json')
    if 'reporting_scope' not in c:
        return root
    if root.name not in ('first_runtime', 'tonight_runtime') or c['reporting_scope'].get('parent_directory') != '..':
        raise ValueError('Unapproved first reporting root or ledger override')
    parent = root.parent
    if sha(parent/'contract.json') != c['reporting_scope']['parent_contract_sha256']:
        raise ValueError('Parent scientific contract changed')
    if root.name == 'tonight_runtime':
        from prepare_oct07_tonight_core import SCOPE
        if c['reporting_scope'].get('id') != SCOPE:
            raise ValueError('Unapproved tonight scope identity')
    return parent


def queue_path(root):
    return parent_root(root)/'queue.sqlite'


def paused(root):
    root = Path(root).resolve()
    parent = parent_root(root)
    paths = {root, parent}
    if any((p/flag).exists() for p in paths for flag in ('PAUSE','STOP')):
        return True
    plan = parent/'first_manuscript_protocol.json'
    if root==parent and plan.is_file():
        # A scoped continuation shares the ledger; it does not lift the user's
        # pause of the60816-row parent. Legacy bundles must also fail closed.
        return read(plan).get('execution_control', {}).get('parent_paused', True) is not False
    return False


def verify_runtime(root):
    root = Path(root).resolve()
    c = read(root/'contract.json')
    if 'reporting_scope' not in c:
        return None
    if root.name == 'tonight_runtime':
        from prepare_oct07_tonight_core import verify
        return verify(root)
    parent = parent_root(root)
    plan = read(parent/'first_manuscript_protocol.json')
    original = read(parent/'contract.json')
    if sha(parent/'matrix.csv') != plan['parent']['matrix_sha256']:
        raise ValueError('Parent matrix changed')
    specs = {s: sha(root/'specifications'/(s+'.specs.jsonl')) for s in ('train','validation','test')}
    expected = scoped_contract(original, plan, plan['matrix_sha256'], specs)
    if c != expected or sha(root/'matrix.csv') != plan['matrix_sha256']:
        raise ValueError('First contract or scientific budgets differ from authorized metadata view')
    with (parent/'first_manuscript_scene_ids.csv').open(encoding='utf-8',newline='') as f:
        selected = list(csv.DictReader(f))
    with (root/'matrix.csv').open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    if dict(Counter(r['run_class']+'/'+r['kind'] for r in rows)) != c['counts']:
        raise ValueError('Derived counts do not describe the frozen first matrix')
    public = [r for r in rows if r['run_class']=='development'
              and r['method'] in c['sources']['author_methods']]
    baseline = [r for r in rows if r['run_class']=='development'
                and r['method'] in (*c['sources']['author_methods'], 'fifo', 'no_comm', 'all_view')]
    if len(public)!=c['baseline_first']['public_rows'] or len(baseline)!=c['baseline_first']['baseline_stage_rows']:
        raise ValueError('First baseline denominators differ from actual rows')
    from prepare_oct07_first_manuscript import select_test
    original_test = [json.loads(line) for line in (parent/'specifications/test.specs.jsonl').read_text(encoding='utf-8').splitlines()]
    if {r['episode_id'] for r in selected if r['split']=='test'} != {s['episode_id'] for s in select_test(original_test)}:
        raise ValueError('First TEST IDs no longer match the prospective metadata-only rule')
    for split, count in [('train',120),('validation',24),('test',48)]:
        declared = {r['episode_id'] for r in selected if r['split']==split}
        actual = [json.loads(line) for line in (root/'specifications'/(split+'.specs.jsonl')).read_text(encoding='utf-8').splitlines()]
        original_specs = {s['episode_id']:s for s in map(json.loads,(parent/'specifications'/(split+'.specs.jsonl')).read_text(encoding='utf-8').splitlines())}
        if len(actual)!=count or len(declared)!=count or {s['episode_id'] for s in actual}!=declared or any(s!=original_specs.get(s['episode_id']) for s in actual):
            raise ValueError('First scene identity differs from predeclared parent subset')
    if (root/'queue.sqlite').exists():
        raise ValueError('A second first-scope task database is prohibited')
    return dict(scope_id=plan['scope_id'], parent_rows=60816, first_rows=7560,
        shared_ledger=str(queue_path(root)), training_fit_limit=24, TEST_scenes=48)


def enforce_prior_first_freeze(root, code=None):
    """A later parent run cannot tune code/first weights after seeing TEST48."""
    root = Path(root).resolve()
    if 'reporting_scope' in read(root/'contract.json'):
        return
    first = root/'first_runtime'
    if not (first/'test_results_release.json').exists():
        return
    from oct07_baseline_freeze import require_test_release
    require_test_release(first,'formal')
    selection = read(first/'algorithm_selection.json')
    code = Path(code or Path(__file__).parent)
    for name, digest in selection['implementation_sha256'].items():
        if Path(name).name != name or sha(code/name)!=digest:
            raise ValueError('Scientific version changed after first TEST release; no TEST-guided tuning')
    for name, digest in selection['fit_identity'].items():
        if sha(root/name)!=digest:
            raise ValueError('First Qwen checkpoint changed after first TEST release')
