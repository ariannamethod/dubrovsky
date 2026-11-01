# Literary translation of Dubrovsky

Translate the question and answer as a piece of authored absurdist writing.
Preserve the character, the progression of associations, the joke, and the
rhythm. Produce natural grammar in the requested language, including case,
gender, number, and agreement. A surreal event is intentional; do not repair
its logic or explain it to the reader.

Preserve first-person and third-person shifts. Keep the speaker and other
agents distinct even when an umbrella becomes mayor or a doubt hires a lawyer.
Keep Alexey Dubrovsky's name recognizable and use one spelling consistently
within each target language. Do not add a generic assistant introduction or
identity disclaimer. Do not turn a short punchline into an explanatory essay.

When a pun cannot survive literally, use a natural target-language turn that
preserves its role in the joke. Preserve deliberate profanity and register.
Do not silently discard sentences, repetitions, or inconvenient implications.
If a passage needs editorial revision rather than translation, flag it for a
separate review instead of rewriting the source's position.

For each supplied record, return its unchanged `source_id`, `split`, and
`target_language`, plus translated `question` and `answer`. Keep commentary
outside those fields. Never move a validation record into the training split;
translations of a source record inherit that source's split.

Russian and Hebrew are the first targets. Review grammatical quality and
literary voice separately. Compare the multilingual body to the original
Huihui model and to the English SFT body on prompts excluded from training
and preference construction.
