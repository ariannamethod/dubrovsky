"""Literal output counters; preserve evidence spans and count per checkpoint/language."""
import argparse
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re
import unicodedata


def words(text):
    """Unicode letter runs, retaining combining marks and exact character offsets."""
    start = None
    for i, char in enumerate(text + ' '):
        letter = char.isalpha()
        mark = unicodedata.category(char).startswith('M')
        if letter or (mark and start is not None):
            if start is None:
                start = i
        elif start is not None:
            yield text[start:i], start, i
            start = None


def diagnostics(text, question):
    tokens = list(words(text))
    mixed = []
    for word, start, end in tokens:
        scripts = {unicodedata.name(c, '').split(' ')[0] for c in word if c.isalpha()}
        if 'LATIN' in scripts and scripts.intersection({'CYRILLIC', 'HEBREW'}):
            mixed.append(dict(text=word, start=start, end=end, scripts=sorted(scripts)))
    first = tokens[0][0] if tokens else ''
    first_plain = ''.join(c for c in unicodedata.normalize('NFD', first)
                          if not unicodedata.category(c).startswith('M'))
    forms = [dict(text=w, start=s, end=e) for w,s,e in tokens
             if w.casefold() in {'подал', 'подала', 'подало', 'подали'}]
    # Full original prompt copied at the start, allowing leading whitespace/quotes.
    left = text.lstrip(' \t\r\n\"«„“')
    full_prompt_echo = bool(question.strip() and left.startswith(question.strip()))
    # Longest literal prompt substring at the answer opening, bounded at a word end.
    normalized_question = re.sub(r'\s+', ' ', question).strip()
    normalized_text = re.sub(r'\s+', ' ', left).strip()
    prefix = ''
    for _, _, end in words(normalized_text):
        candidate = normalized_text[:end]
        if candidate in normalized_question:
            prefix = candidate
        else:
            break
    echo_words = len(list(words(prefix)))
    q_words = [w.casefold() for w,_,_ in words(question)]
    opening = tokens[:24]
    match = SequenceMatcher(None, [w.casefold() for w,_,_ in opening], q_words,
                            autojunk=False).find_longest_match()
    shared = (text[opening[match.a][1]:opening[match.a+match.size-1][2]]
              if match.size else '')
    body_match = SequenceMatcher(None, [w.casefold() for w,_,_ in tokens], q_words,
                                 autojunk=False).find_longest_match()
    body_shared = (text[tokens[body_match.a][1]:tokens[body_match.a+body_match.size-1][2]]
                   if body_match.size else '')
    return dict(mixed_script_words=mixed, podal_forms=forms,
                opening_tokens=[w for w,_,_ in tokens[:3]],
                hebrew_shenit_opening=first_plain in {'שנית', 'שניתן'},
                full_prompt_echo=full_prompt_echo,
                opening_prompt_overlap=dict(text=shared, words=match.size,
                                            flagged=match.size >= 6),
                answer_prompt_overlap=dict(text=body_shared, words=body_match.size,
                                           flagged=body_match.size >= 6),
                prompt_prefix_echo=dict(text=prefix, words=echo_words,
                                        flagged=echo_words >= 6))


def summarize(rows):
    groups = defaultdict(list)
    details = []
    for row in rows:
        messages = row.get('messages', [])
        question = next((m['content'] for m in reversed(messages) if m['role']=='user'), '')
        result = diagnostics(row['text'], question)
        item = dict(model=row['model'], language=row['language'], case_id=row['case_id'],
                    seed=row.get('seed'), configuration_id=row.get('configuration_id'),
                    turn=row.get('turn'), diagnostics=result)
        details.append(item)
        groups[(row['model'],row['language'],row.get('seed'),row.get('configuration_id'))].append(result)
    totals = []
    for (model, language, seed, configuration), items in sorted(groups.items(),key=lambda x:repr(x[0])):
        counts = dict(mixed_script_answers=sum(bool(x['mixed_script_words']) for x in items),
                      podal_answers=sum(bool(x['podal_forms']) for x in items),
                      hebrew_shenit_openings=sum(x['hebrew_shenit_opening'] for x in items),
                      full_prompt_echoes=sum(x['full_prompt_echo'] for x in items),
                      opening_prompt_overlaps=sum(x['opening_prompt_overlap']['flagged'] for x in items),
                      answer_prompt_overlaps=sum(x['answer_prompt_overlap']['flagged'] for x in items),
                      long_prompt_prefix_echoes=sum(x['prompt_prefix_echo']['flagged'] for x in items))
        openings = Counter(tuple(x['opening_tokens'][:2]) for x in items)
        totals.append(dict(model=model, language=language, seed=seed, configuration_id=configuration,
                           answers=len(items), **counts,
                           leading_pairs=[dict(tokens=list(k),count=v) for k,v in openings.most_common(8)]))
    return dict(groups=totals, details=details)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('inputs', type=Path, nargs='+')
    p.add_argument('--output', type=Path, required=True)
    a=p.parse_args()
    rows=[json.loads(line) for path in a.inputs for line in path.read_text().splitlines() if line.strip()]
    result=summarize(rows)
    result['sources']=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in a.inputs]
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
