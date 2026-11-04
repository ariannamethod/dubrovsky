"""Validate a complete weight probe and compare paired sampling-seed slots."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from compare_sampling import validate_messages


def summarize(spec, rows, spec_sha256):
    if not rows or rows[0].get('kind') != 'provenance':
        raise ValueError('Missing provenance')
    if rows[0]['spec_sha256'] != spec_sha256:
        raise ValueError('Specification hash differs')
    if rows[-1].get('kind') != 'complete':
        raise ValueError('Incomplete probe')
    restorations = [r for r in rows if r['kind'] == 'restoration']
    if len(restorations) != 1 or not all(restorations[0].get(k) for k in
            ('edited_tensors_restored', 'other_parameters_unchanged')):
        raise ValueError('Missing successful restoration')
    if restorations[0].get('unchanged_parameter_sha256') != rows[0]['unchanged_parameter_sha256']:
        raise ValueError('Unedited parameter digest changed')
    targets = {t['id']: t for t in spec['targets']}
    target = targets.get(rows[0]['target']['id'])
    if rows[0]['target'] != target:
        raise ValueError('Checkpoint provenance differs')
    variants = [v['id'] for v in spec['variants']]
    installed = [r for r in rows if r['kind'] == 'installed']
    if len(installed) != len(variants) or {r['variant']['id'] for r in installed} != set(variants):
        raise ValueError('Missing, extra or duplicate installed variants')
    for row in installed:
        if row['tensor_sha256'] != target['variant_sha256'][row['variant']['id']]:
            raise ValueError('Installed tensor hashes differ')
    cases = {c['id']: c for c in spec.get('generations', [])}
    expected = {(v, c['id'], seed): slot for v in variants for c in cases.values()
                for slot, seed in enumerate(c.get('seeds', spec['seeds']))}
    generations = [r for r in rows if r['kind'] == 'generation']
    observed = [(r['variant'], r['case_id'], r['seed']) for r in generations]
    if len(observed) != len(set(observed)) or set(observed) != set(expected):
        raise ValueError('Missing, extra or duplicate generation cells')
    dialogues = {c['id']: c for c in spec.get('dialogues', [])}
    dialogue_rows = [r for r in rows if r['kind'] == 'dialogue']
    expected_turns = {(v, c['id'], seed, turn): slot for v in variants for c in dialogues.values()
        for slot, seed in enumerate(c.get('seeds', spec['seeds']))
        for turn in range(1, len(c['user_turns'])+1)}
    observed_turns = [(r['variant'], r['case_id'], r['seed'], r['turn']) for r in dialogue_rows]
    if len(observed_turns) != len(set(observed_turns)) or set(observed_turns) != set(expected_turns):
        raise ValueError('Missing, extra or duplicate dialogue turns')
    histories = {}
    for row in sorted(dialogue_rows, key=lambda r: (r['variant'], r['case_id'], r['seed'], r['turn'])):
        key = (row['variant'], row['case_id'], row['seed'])
        question = dialogues[row['case_id']]['user_turns'][row['turn']-1]
        messages = validate_messages(histories.get(key, []) + [dict(role='user', content=question)])
        if row['messages'] != messages or row['turn_seed'] != row['seed'] + row['turn'] - 1:
            raise ValueError('Dialogue history or turn seed differs')
        histories[key] = messages + [dict(role='assistant', content=row['text'])]
    contrasts = [r for r in rows if r['kind'] == 'contrast']
    observed_pairs = [(r['variant'], r['case_id']) for r in contrasts]
    expected_pairs = {(v, c['id']) for v in variants for c in spec.get('contrasts', [])}
    if len(observed_pairs) != len(set(observed_pairs)) or set(observed_pairs) != expected_pairs:
        raise ValueError('Missing, extra or duplicate contrast cells')
    contrast_cases = {c['id']: c for c in spec.get('contrasts', [])}
    for row in contrasts:
        case = contrast_cases[row['case_id']]
        if (validate_messages(row['messages']) != validate_messages(case['messages']) or
                row['branch_texts'] != case['branches'] or
                set(row['branches']) != set(case['branches']) or
                row['language'] != case['language'] or
                row['enable_thinking'] is not False or
                case.get('enable_thinking', False) is not False):
            raise ValueError('Contrast prompt, continuations or mode differs')
    def validate_decoding(row, case):
        if (row['sampling'] != spec['sampling'] or
                row['max_new_tokens'] != case.get('max_new_tokens', spec['max_new_tokens'])):
            raise ValueError('Recorded decoding settings differ')
    grouped = defaultdict(list)
    for row, key in zip(generations, observed):
        case = cases[row['case_id']]
        validate_decoding(row, case)
        if row['language'] != case['language'] or row['enable_thinking'] != case.get('enable_thinking', False):
            raise ValueError('Generation case metadata differs')
        if validate_messages(row['messages']) != validate_messages(case['messages']):
            raise ValueError('Fixed history differs')
        grouped[('fixed', row['variant'], expected[key], row['language'], row['enable_thinking'])].append(row)
    for row, key in zip(dialogue_rows, observed_turns):
        case = dialogues[row['case_id']]
        validate_decoding(row, case)
        if row['language'] != case['language'] or row['enable_thinking'] != case.get('enable_thinking', False):
            raise ValueError('Dialogue case metadata differs')
        grouped[('dialogue', row['variant'], expected_turns[key], row['language'], row['enable_thinking'])].append(row)
    summaries = []
    for (mode, variant, slot, language, thinking), group in sorted(grouped.items()):
        summaries.append(dict(mode=mode, variant=variant, seed_slot=slot, language=language,
            enable_thinking=thinking, answers=len(group),
            exact_copies=sum(bool(r['history_diagnostics']['exact_prior_message_indices']) for r in group),
            prior_word4_fraction_mean=sum(r['history_diagnostics']['prior_word4_fraction'] for r in group)/len(group),
            eos=sum(r['eos_reached'] for r in group),
            mixed_script=sum(bool(r['literal_diagnostics']['mixed_script_words']) for r in group),
            cjk=sum(bool(r['literal_diagnostics']['cjk_spans']) for r in group),
            role_frame=sum(bool(r['literal_diagnostics']['role_performance_mentions']) for r in group),
            latin_containing=sum(bool(r['literal_diagnostics'].get('latin_spans')) for r in group),
            podal_forms=sum(bool(r['literal_diagnostics'].get('podal_forms')) for r in group),
            sh_initial_candidates=sum(bool(r['literal_diagnostics'].get('hebrew_sh_initial_candidate')) for r in group),
            shenit_openings=sum(bool(r['literal_diagnostics'].get('hebrew_shenit_opening')) for r in group),
            full_prompt_echoes=sum(bool(r['literal_diagnostics'].get('full_prompt_echo')) for r in group),
            empty_thinking_closed=sum(r['empty_thinking_closed'] for r in group),
            token_count_mean=sum(len(r['token_ids']) for r in group)/len(group)))
    margins = {(r['variant'], r['case_id']):
        r['branches']['productive']['mean_logp'] - r['branches']['repeat']['mean_logp']
        for r in contrasts}
    return dict(target=rows[0]['target']['id'], spec_sha256=spec_sha256,
        generations=len(generations), dialogue_turns=len(dialogue_rows), contrast_pairs=len(contrasts),
        integrity_verified=True, by_seed=summaries,
        contrasts=[dict(variant=v, case_id=c, productive_minus_repeat=m,
                       change_from_baseline=m-margins[('baseline', c)])
                   for (v, c), m in margins.items()])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    spec_bytes = args.spec.read_bytes()
    rows = [json.loads(line) for line in args.input.read_text().splitlines()]
    result = summarize(json.loads(spec_bytes), rows, hashlib.sha256(spec_bytes).hexdigest())
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
