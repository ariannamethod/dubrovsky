"""Native-template token provenance for the user-only memory writer."""
from sft import conversation_messages, encode


def isolated_user_stream(tokenizer, messages, max_length):
    """Native user-role serialization with no assistant/system content or padding."""
    users = [dict(role='user', content=m['content']) for m in messages if m['role']=='user']
    if not users:
        raise ValueError('User-only stream cannot be empty')
    rendered = tokenizer.apply_chat_template(users, tokenize=False,
        add_generation_prompt=False, enable_thinking=False)
    ids = tokenizer.apply_chat_template(users, tokenize=True,
        add_generation_prompt=False, enable_thinking=False, return_dict=False)
    if len(ids) > max_length:
        raise ValueError('User-only stream exceeds context limit; refusing truncation')
    return dict(memory_input_ids=ids,
                memory_user_mask=user_content_mask(tokenizer, users, rendered, ids))


def memory_inputs(model, row, device):
    """Convert the persisted source stream only for format-2 checkpoints."""
    import torch
    if model.expert_config.format_version == 1:
        return {}
    return dict(memory_input_ids=torch.tensor([row['memory_input_ids']], device=device),
                memory_user_mask=torch.tensor([row['memory_user_mask']], dtype=torch.bool, device=device))


def user_content_mask(tokenizer, messages, rendered, ids):
    if not tokenizer.is_fast:
        raise ValueError('A fast tokenizer with character offsets is required')
    spans = []
    for index, message in enumerate(messages):
        if message['role'] != 'user':
            continue
        cumulative = tokenizer.apply_chat_template(messages[:index+1], tokenize=False,
            add_generation_prompt=False, enable_thinking=False)
        if not rendered.startswith(cumulative):
            raise ValueError('Chat template rewrites earlier messages')
        header = '<|im_start|>user\n'
        suffix = header + message['content'] + '<|im_end|>\n'
        if not cumulative.endswith(suffix):
            raise ValueError('Unexpected native Qwen user serialization')
        start = len(cumulative) - len(suffix) + len(header)
        spans.append((start, start + len(message['content'])))
    tokenized = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    if tokenized['input_ids'] != ids:
        raise ValueError('Token offsets differ from the native chat template IDs')
    # A boundary-spanning token cannot write template/assistant text into the bank.
    return [b > a and any(start <= a and b <= end for start, end in spans)
            for a, b in tokenized['offset_mapping']]


def encode_expert_row(tokenizer, row, max_length):
    encoded = encode(tokenizer, row, max_length)
    messages = conversation_messages(row) + [dict(role='assistant', content=row['answer'],
                                                  reasoning_content='')]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False,
        add_generation_prompt=False, enable_thinking=False)
    encoded['user_mask'] = user_content_mask(tokenizer, messages, rendered, encoded['input_ids'])
    encoded.update(isolated_user_stream(tokenizer, messages, max_length))
    encoded['id'] = row['id']
    encoded['language'] = row.get('language', 'en')
    if not any(encoded['user_mask']):
        raise ValueError('Training example has no writable user content')
    return encoded


def encode_expert_prompt(tokenizer, row, max_length):
    messages = conversation_messages(row)
    rendered = tokenizer.apply_chat_template(messages, tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer.apply_chat_template(messages, tokenize=True,
        add_generation_prompt=True, enable_thinking=False, return_dict=False)
    if len(ids) > max_length:
        raise ValueError('Full conversation exceeds the context limit; refusing truncation')
    return dict(input_ids=ids, user_mask=user_content_mask(tokenizer, messages, rendered, ids),
                **isolated_user_stream(tokenizer, messages, max_length))


def validate_expert_splits(train, validation):
    import json
    def contexts(rows):
        return {json.dumps([r.get('history', []), r['question']], ensure_ascii=False,
                           sort_keys=True) for r in rows}
    if contexts(train) & contexts(validation):
        raise ValueError('Train and validation share a conversation prefix')
    families = [{r['family'] for r in rows if r.get('family')} for rows in (train, validation)]
    if families[0] & families[1]:
        raise ValueError('Counterfactual dialogue families must stay in one split')
    for rows in (train, validation):
        if {r.get('language', 'en') for r in rows} != {'en', 'ru'}:
            raise ValueError('This pilot requires both English and Russian in each split')
