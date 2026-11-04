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


def script_spans(text, prefixes):
    """Literal character runs, including independent foreign words and mixed words."""
    hits = []
    start = None
    for i, char in enumerate(text + ' '):
        matching = any(unicodedata.name(char, '').startswith(p) for p in prefixes)
        if matching and start is None:
            start = i
        elif not matching and start is not None:
            hits.append(dict(text=text[start:i], start=start, end=i))
            start = None
    return hits


def role_mentions(text):
    """Candidate performance-frame phrases; scene-aware reading determines errors."""
    patterns = {
        'he': r'(?<!\w)אני\s+(?:מגלמת|מגלם|משחקת|משחק)(?!\w)',
        'ru': r'(?<!\w)я\s+(?:играю\s+роль|изображаю|исполняю\s+роль)(?!\w)',
        'en': r"(?<!\w)I\s+(?:am|'m)\s+(?:playing|portraying|roleplaying|role-playing)(?!\w)",
    }
    return [dict(language=language, text=m.group(), start=m.start(), end=m.end())
            for language, pattern in patterns.items()
            for m in re.finditer(pattern, text, re.IGNORECASE)]


def answer_history_diagnostics(text, messages):
    """Compare only with assistant answers actually supplied in this row's history."""
    history = list(messages)
    if history and history[-1]['role'] == 'assistant':
        if history[-1]['content'] != text:
            raise ValueError('Saved final assistant message differs from answer text')
        history = history[:-1]
    previous = [(i, m['content']) for i, m in enumerate(history) if m['role'] == 'assistant']
    def grams(value):
        tokens = [w.casefold() for w, _, _ in words(value)]
        return [tuple(tokens[i:i+4]) for i in range(max(0, len(tokens)-3))]
    current = grams(text)
    prior = {g for _, answer in previous for g in grams(answer)}
    return dict(previous_assistant_messages=len(previous),
                exact_prior_message_indices=[i for i, answer in previous if text and answer == text],
                prior_word4_fraction=sum(g in prior for g in current)/len(current) if current else 0.,
                current_word4_count=len(current))


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
                latin_spans=script_spans(text, ('LATIN ',)),
                cjk_spans=script_spans(text, ('CJK ', 'HIRAGANA ', 'KATAKANA ', 'HANGUL ')),
                role_performance_mentions=role_mentions(text),
                opening_tokens=[w for w,_,_ in tokens[:3]],
                hebrew_shenit_opening=first_plain in {'שנית', 'שניתן'},
                # A screen for manual reading, not a Hebrew clause parser: many
                # independent words (שלום, שוב) also begin with this character.
                hebrew_sh_initial_candidate=first_plain.startswith('ש'),
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
        result['answer_history'] = answer_history_diagnostics(row['text'], messages)
        item = dict(model=row['model'], language=row['language'], case_id=row['case_id'],
                    seed=row.get('seed'), configuration_id=row.get('configuration_id'),
                    turn=row.get('turn'), diagnostics=result)
        details.append(item)
        groups[(row['model'],row['language'],row.get('seed'),row.get('configuration_id'))].append(result)
    totals = []
    for (model, language, seed, configuration), items in sorted(groups.items(),key=lambda x:repr(x[0])):
        counts = dict(mixed_script_answers=sum(bool(x['mixed_script_words']) for x in items),
                      latin_containing_answers=sum(bool(x['latin_spans']) for x in items),
                      cjk_containing_answers=sum(bool(x['cjk_spans']) for x in items),
                      role_performance_answers=sum(bool(x['role_performance_mentions']) for x in items),
                      exact_prior_answer_reuses=sum(bool(x['answer_history']['exact_prior_message_indices']) for x in items),
                      hebrew_sh_initial_candidates=sum(x['hebrew_sh_initial_candidate'] for x in items),
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
