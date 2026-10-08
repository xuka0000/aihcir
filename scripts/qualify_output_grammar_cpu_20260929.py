"""Bounded CPU-only qualification. Never loads weights or starts inference.

Run --fetch once (two transports/file maximum); --check is entirely offline.
The grammar is over generated suffix IDs, never full prompt IDs.
"""
import argparse
import hashlib
import importlib.metadata
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'evidence/output_grammar_cpu_20260929_v1'
TASK = 'output-grammar-cpu-builder-20260929'
REPORT_MILESTONES = False
MODELS = {
    'qwen3': ('Qwen/Qwen3-VL-8B-Instruct', '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b',
              'evidence/qwen3_backbone_20260918_v1/official_model_metadata.json',
              'evidence/qwen3_backbone_20260918_v1/modelscope_metadata.json', 'chat_template.json'),
    'internvl35_hf': ('OpenGVLab/InternVL3_5-8B-HF', '741a7d03020411e666c6109218ab71e08151ef86',
                     'evidence/internvl_dev231_20260928_v1/sources/hf_metadata.json',
                     'evidence/internvl_dev231_20260928_v1/sources/modelscope_metadata.json', 'chat_template.jinja'),
}


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def write(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def report(completed, step):
    if not REPORT_MILESTONES:
        return
    path = OUT / ('progress_m%d.json' % completed)
    write(path, dict(total=3, completed=completed, succeeded=completed, failed=0,
                     running=int(completed < 3), pending=max(0, 2-completed), accepted=0,
                     unit='bounded deliverable milestone'))
    result = subprocess.run([sys.executable, '-X', 'utf8', str(ROOT/'scripts/team_runtime.py'), 'report',
                             '--task', TASK, '--progress', str(path), '--current-step', step],
                            capture_output=True, text=True, encoding='utf-8')
    (OUT / ('report_m%d_%d.log' % (completed,time.time_ns()))).write_text(result.stdout+result.stderr, encoding='utf-8')
    if result.returncode:
        raise RuntimeError('Team report failed; receipt retained')


def verify(path, entry):
    data = path.read_bytes()
    blob = hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    expected = entry.get('lfs', {}).get('sha256')
    ok = len(data) == entry['size'] and (hashlib.sha256(data).hexdigest() == expected if expected else blob == entry['blobId'])
    if not ok:
        raise ValueError('Official size/gitblob mismatch: '+path.name)
    return dict(size=len(data), sha256=sha(path), gitblob_sha1=blob,
                official_gitblob=entry['blobId'], verified=True,
                verification_basis='LFS object SHA256 and size' if expected else 'Git blob SHA1 and size',
                official_lfs_sha256=expected,
                official_lfs_pointer_gitblob=entry['blobId'] if expected else None)


def fetch():
    packet = read(ROOT / ('.codex/team_runtime/'+TASK+'.json'))
    receipts_path = OUT/'fetch_receipts.json'
    receipts = read(receipts_path) if receipts_path.exists() else []
    provenance = {}
    for key, (repo, revision, metadata_path, mirror_path, template) in MODELS.items():
        metadata = read(ROOT/metadata_path)
        mirror = {r['Path']: r for r in read(ROOT/mirror_path)['Data']['Files']}
        assert metadata['sha'] == revision and metadata['id'] == repo
        entries = {r['rfilename']: r for r in metadata['siblings']}
        directory = OUT/'sources'/key
        directory.mkdir(parents=True, exist_ok=True)
        write(directory/'official_metadata.json', metadata)
        sources = {}
        for name in ['tokenizer.json', 'tokenizer_config.json', 'generation_config.json', template]:
            entry = entries[name]
            assert entry['size'] <= 15_000_000
            dest = directory/name
            if dest.exists():
                sources[name] = dict(path=str(dest.relative_to(ROOT)), **verify(dest, entry))
                continue
            prior = sum(r['model'] == key and r['file'] == name for r in receipts)
            urls = ['https://huggingface.co/'+repo+'/resolve/'+revision+'/'+name,
                    'https://modelscope.cn/api/v1/models/'+repo+'/repo?'+
                    urlencode(dict(Revision=mirror[name]['Revision'], FilePath=name))]
            for attempt in range(prior, 2):
                if time.time()+32 >= packet['deadline']:
                    raise TimeoutError('Task deadline: no further fetch')
                partial = directory/(name+'.partial')
                started = time.time()
                row = dict(model=key, file=name, attempt=attempt+1, url=urls[attempt],
                           started_unix=started, max_seconds=30, max_bytes=15_000_000)
                try:
                    result = subprocess.run(['curl.exe','--fail','--location','--silent','--show-error',
                        '--connect-timeout','10','--max-time','30','--max-filesize','15000000',
                        '--output',str(partial),urls[attempt]], capture_output=True, timeout=32)
                    if result.returncode:
                        # Do not retain signed redirect addresses in stderr.
                        raise RuntimeError('curl exit '+str(result.returncode))
                    checked = verify(partial, entry)
                    partial.replace(dest)
                    sources[name] = dict(path=str(dest.relative_to(ROOT)), **checked)
                    row.update(status='verified', **checked)
                except Exception as error:
                    row.update(status='failed', error_type=type(error).__name__, error=str(error))
                row['elapsed_seconds'] = time.time()-started
                receipts.append(row)
                write(receipts_path, receipts)
                if row['status'] == 'verified':
                    break
            if name not in sources:
                raise RuntimeError('Two bounded attempts exhausted: '+key+'/'+name)
        provenance[key] = dict(repository=repo, revision=revision, files=sources,
            official_metadata_source=metadata_path, official_metadata_source_sha256=sha(ROOT/metadata_path),
            historical_runtime_file_identity=None,
            historical_unknown_reason='Downloaded pinned files are verified; historical runtime file bytes were not retained.')
    write(OUT/'provenance.json', provenance)
    report(1, 'Pinned files and actual template source verified; CPU checks next')
    print('milestone1: pinned source gitblobs verified', flush=True)


class SuffixTrie:
    def __init__(self, sequences, eos):
        self.sequences = {tuple(s) for s in sequences}
        if not self.sequences or any(not s or eos in s for s in self.sequences):
            raise ValueError('Empty answer or EOS inside answer')
        self.eos = eos

    def allowed(self, suffix):
        suffix = tuple(suffix)
        allowed = {s[len(suffix)] for s in self.sequences
                   if len(s) > len(suffix) and s[:len(suffix)] == suffix}
        if suffix in self.sequences:
            allowed.add(self.eos)
        return sorted(allowed)

    def callback(self, prompt_length):
        def prefix_allowed_tokens_fn(batch_id, ids):
            ids = ids.tolist() if hasattr(ids, 'tolist') else list(ids)
            if len(ids) < prompt_length:
                raise ValueError('Input shorter than fixed prompt length')
            return self.allowed(ids[prompt_length:])
        return prefix_allowed_tokens_fn

    def accepts(self, ids):
        return bool(ids) and ids[-1] == self.eos and tuple(ids[:-1]) in self.sequences


def exhaustive_graph_check(trie, vocab_size):
    # Every reachable state checks EVERY vocabulary ID; complement tokens have no edge.
    prefixes = {s[:i] for s in trie.sequences for i in range(len(s)+1)}
    transitions = []
    checked = 0
    for prefix in sorted(prefixes):
        allowed = trie.allowed(prefix)
        expected = {s[len(prefix)] for s in trie.sequences
                    if len(s)>len(prefix) and s[:len(prefix)] == prefix}
        if prefix in trie.sequences:
            expected.add(trie.eos)
        for token in range(vocab_size):
            assert (token in allowed) == (token in expected)
            checked += 1
        if prefix not in trie.sequences:
            assert trie.eos not in allowed
        transitions.append(dict(prefix=list(prefix), allowed=allowed,
                                rejected_vocabulary_ids=vocab_size-len(allowed)))
    assert trie.allowed([trie.eos]) == [] and not trie.accepts([trie.eos])
    return dict(reachable_states=len(prefixes), vocabulary_transition_checks=checked,
                transitions=transitions, accepted_paths=[list(s)+[trie.eos] for s in sorted(trie.sequences)],
                proof='Finite trie admits only listed answer sequences then EOS; all other vocabulary transitions absent.')


def check():
    dependencies = OUT/'dependencies'
    if dependencies.exists():
        sys.path.insert(0,str(dependencies))
    from tokenizers import Tokenizer
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    from jinja2 import StrictUndefined
    original_paths = ['evidence/baseline_first_20260928_v1/predictions.jsonl',
        'evidence/baseline_first_20260928_v1/evaluation.json',
        'evidence/internvl_dev231_20260928_v1/predictions.jsonl',
        'evidence/internvl_dev231_20260928_v1/evaluation.json',
        'evidence/internvl_recovery_20260929_v1/sensitivity.json',
        'scripts/run_aircop_complete_development.py', 'scripts/run_internvl_complete_development.py']
    original_hashes = {p:sha(ROOT/p) for p in original_paths}
    provenance = read(OUT/'provenance.json')
    episodes = read(ROOT/'evidence/baseline_first_20260928_v1/assets/public_episodes.json')
    results = {}
    for key, (_, _, _, _, template_name) in MODELS.items():
        directory = OUT/'sources'/key
        metadata = read(directory/'official_metadata.json')
        entries = {r['rfilename']:r for r in metadata['siblings']}
        for name in provenance[key]['files']:
            verify(directory/name, entries[name])
        tokenizer = Tokenizer.from_file(str(directory/'tokenizer.json'))
        tokenizer.no_padding()
        tokenizer.no_truncation()
        config = read(directory/'tokenizer_config.json')
        for token_id, added in config.get('added_tokens_decoder',{}).items():
            assert tokenizer.token_to_id(added['content']) == int(token_id)
        gen = read(directory/'generation_config.json')
        template = (read(directory/template_name)['chat_template'] if template_name.endswith('.json')
                    else (directory/template_name).read_text(encoding='utf-8'))
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
            undefined=StrictUndefined, extensions=['jinja2.ext.loopcontrols'])
        def raise_exception(message):
            raise ValueError(message)
        env.globals['raise_exception'] = raise_exception
        compiled = env.from_string(template)
        eos_ids = gen['eos_token_id']
        eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
        eos_token = config['eos_token']
        eos_token = eos_token['content'] if isinstance(eos_token,dict) else eos_token
        eos = tokenizer.token_to_id(eos_token)
        assert eos in eos_ids
        contexts = []
        for episode in episodes:
            question = episode['question']+'\n'+'\n'.join(k+'. '+v for k,v in episode['options'].items())
            question += "\nAnswer with the option's letter from the given choices directly."
            content = [dict(type='text', text=question)]+[dict(type='image') for _ in episode['views']]
            prompt = compiled.render(messages=[dict(role='user',content=content)],
                add_generation_prompt=True, tools=None, add_vision_id=False,
                **{k:v for k,v in config.items() if k in ['bos_token','eos_token']})
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
            assert prompt.endswith('<|im_start|>assistant\n')
            image_marker = '<|image_pad|>' if key=='qwen3' else '<IMG_CONTEXT>'
            assert prompt.count(image_marker) == len(episode['views'])
            answer_ids = {}
            for letter in 'ABCD':
                full_ids = tokenizer.encode(prompt+letter, add_special_tokens=False).ids
                assert full_ids[:len(prompt_ids)] == prompt_ids, 'Boundary merges prompt with answer'
                suffix = full_ids[len(prompt_ids):]
                assert suffix and tokenizer.decode(suffix, skip_special_tokens=False) == letter
                assert tokenizer.decode(prompt_ids+suffix, skip_special_tokens=False) == prompt+letter
                answer_ids[letter] = suffix
            trie = SuffixTrie(answer_ids.values(), eos)
            callback = trie.callback(len(prompt_ids))
            assert callback(0, prompt_ids) == trie.allowed([])
            assert eos not in callback(0, prompt_ids)
            for letter, suffix in answer_ids.items():
                for i in range(len(suffix)):
                    assert suffix[i] in callback(0,prompt_ids+suffix[:i])
                assert callback(0,prompt_ids+suffix) == [eos]
                assert callback(0,prompt_ids+suffix+[eos]) == []
                assert trie.accepts(suffix+[eos])
            contexts.append(dict(episode_id=episode['id'], prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                prompt_token_count=len(prompt_ids), image_count=len(episode['views']), answer_suffix_ids=answer_ids,
                prompt_token_prefix_stable=True))
            if len(contexts)==1:
                (directory/'representative_rendered_prompt.txt').write_text(prompt, encoding='utf-8')
        unique = {json.dumps(c['answer_suffix_ids'], sort_keys=True) for c in contexts}
        assert len(unique)==1
        grammar = SuffixTrie(contexts[0]['answer_suffix_ids'].values(), eos)
        graph = exhaustive_graph_check(grammar, tokenizer.get_vocab_size(with_added_tokens=True))
        invalid = ['', ' ', '\n', '\t','A ', ' A','A\n','A.','AB','a','Answer: A',
                   'A because it matches','<think></think>A','<think>A</think>','<|im_end|>',
                   '<|im_start|>assistant\nA']
        for whitespace in [' ', '\t','\n','\r','\u00a0','\u2003']:
            invalid.extend(whitespace+l for l in 'ABCD')
            invalid.extend(l+whitespace for l in 'ABCD')
        rejection = []
        for text in sorted(set(invalid)):
            ids = tokenizer.encode(text, add_special_tokens=False).ids
            accepted = grammar.accepts(ids+[eos])
            assert not accepted
            rejection.append(dict(text=text, token_ids=ids, accepted=False))
        results[key] = dict(status='passed', contexts_checked=len(contexts), contexts=contexts,
            eos_token=eos_token, eos_id=eos, original_generation_eos_ids=eos_ids,
            grammar_eos_policy='Only tokenizer eos_token ID; other original EOS IDs are forbidden unless same ID',
            graph_check=graph, rejected_examples=rejection,
            template_file_sha256=sha(directory/template_name),
            template_render_implementation='Jinja ImmutableSandboxedEnvironment; pinned source bytes; CPU text/image-placeholder rendering only',
            actual_processor_image_token_expansion_checked=False,
            historical_runtime_equivalence=None)
    # Real models may use one token per letter; synthetic fixture tests shared-prefix multi-token branches.
    fixture = SuffixTrie([[7,8],[7,9],[10,11,12],[10,11,13]], 99)
    assert fixture.allowed([])==[7,10] and fixture.allowed([7])==[8,9]
    assert fixture.allowed([10,11])==[12,13] and fixture.allowed([10])==[11]
    assert fixture.allowed([7,8])==[99] and fixture.allowed([7,99])==[]
    graph = exhaustive_graph_check(fixture, 100)
    for sequence in itertools.product([7,8,9,10,11,12,13,99], repeat=4):
        assert fixture.accepts(sequence) == (sequence in [(10,11,12,99),(10,11,13,99)])
    write(OUT/'cpu_checks.json', dict(models=results, synthetic_multitoken_fixture=graph,
        observed_unix=time.time(), versions={x:importlib.metadata.version(x) for x in ['tokenizers','jinja2']},
        weights_loaded=False, model_calls=0, gpu_used=False))
    report(2, 'Both pinned tokenizer contextual generated-suffix tries passed CPU checks')
    print('milestone2: CPU contextual grammar passed for both models', flush=True)
    grammar_parameters = {k:dict(answer_suffix_ids=v['contexts'][0]['answer_suffix_ids'],eos_id=v['eos_id'])
                          for k,v in results.items()}
    grammar_hash = hashlib.sha256(json.dumps(grammar_parameters,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    final_hashes = {p:sha(ROOT/p) for p in original_paths}
    assert final_hashes == original_hashes
    write(OUT/'original_artifact_preservation.json',dict(before=original_hashes,after=final_hashes,
        byte_identity_preserved=True, scientific_scores_recomputed=False))
    protocol = dict(schema='symmetric_generated_suffix_grammar_v1', allowed_decoded_answers=list('ABCD'),
        grammar_parameters=grammar_parameters, grammar_parameters_sha256=grammar_hash,
        grammar_implementation_sha256=sha(Path(__file__)),
        language='Exactly one ASCII answer A/B/C/D followed by tokenizer EOS; no whitespace, explanation, thinking or empty output',
        boundary='Slice generation IDs at fixed actual input_ids.shape[1], including padding and processor-expanded image tokens; never use rendered-text token count at runtime',
        callback='SuffixTrie.callback(prompt_length); reject invalid prefixes with empty allowed set',
        construction='Tokenize prompt+letter; verify complete prompt token prefix stable; extract suffix; verify decoded suffix exactly letter; build finite trie',
        eos='EOS forbidden at root and incomplete branches; sole EOS allowed only after complete answer; generation stops on EOS',
        generation_override='Explicitly set generate(eos_token_id=verified tokenizer EOS ID) for BOTH models; Qwen original secondary EOS151643 forbidden by trie and removed from stop override',
        batch='Future smoke/full runner must use one episode per generate call; fixed prompt_length captured after actual processor',
        token_cap='Set max_new_tokens >= longest answer suffix length+1; retain complete raw IDs, EOS/finish reason/cap/time flags',
        symmetry='Identical decoded language and callback implementation for BOTH models; only verified tokenizer IDs differ',
        model_sources=provenance, immutable_originals='All original predictions, strict scores and exploratory sensitivity remain untouched; future constrained outcomes are a separate full 2x231 matrix',
        future_gate='No inference authorization here. Future exact Transformers4.57.1 processor/generate smoke and explicit new GPU budget/deadline still required.',
        unknowns=['Historical runtime tokenizer/template bytes not retained; equivalence unknown',
            'CPU text rendering does not test actual multimodal processor expansion or Transformers generate callback execution',
            'Answer quality, recovered accuracy and scientific/user acceptance not established'],
        checks_source='evidence/output_grammar_cpu_20260929_v1/cpu_checks.json')
    write(OUT/'protocol.json',protocol)
    write(OUT/'qualification.json',dict(schema='pinned_output_grammar_cpu_qualification_v1',
        status='cpu_qualified_future_runtime_smoke_required', observed_unix=time.time(),
        evidence={p:sha(OUT/p) for p in ['provenance.json','cpu_checks.json','protocol.json','fetch_receipts.json',
                                      'original_artifact_preservation.json']},
        script_sha256=sha(Path(__file__)), model_contexts={k:v['contexts_checked'] for k,v in results.items()},
        dependency_receipt_sha256=sha(OUT/'dependency_receipt.json'),
        failed_fetch_attempts=sum(r['status']=='failed' for r in read(OUT/'fetch_receipts.json')),
        retained_failed_execution_receipts=[str(p.relative_to(ROOT)) for p in sorted(OUT.glob('failure_*.json'))],
        historical_runtime_equivalence=None, scientific_acceptance=None,user_acceptance=None,
        reproduced_command='python scripts/qualify_output_grammar_cpu_20260929.py --check',
        no_new_experiments=True, gpu_budget_reuse=False, original_predictions_modified=False))
    report(3, 'CPU qualification and symmetric protocol delivered; awaiting Evaluator review')
    print('milestone3: protocol and qualification written', flush=True)


def main():
    global REPORT_MILESTONES
    parser = argparse.ArgumentParser()
    parser.add_argument('--fetch',action='store_true')
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--report-milestones',action='store_true',help='First execution only; offline rechecks do not regress applied progress')
    args=parser.parse_args()
    REPORT_MILESTONES = args.report_milestones
    OUT.mkdir(parents=True,exist_ok=True)
    try:
        if args.fetch:
            fetch()
        if args.check:
            check()
    except Exception as error:
        write(OUT/('failure_%d.json'%time.time_ns()),dict(observed_unix=time.time(),
            error_type=type(error).__name__, error=str(error), fetch=args.fetch, check=args.check))
        raise


if __name__=='__main__':
    main()
