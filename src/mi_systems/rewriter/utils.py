import spacy

# Disable the advanced NLP features in the pipeline for efficiency.
nlp = spacy.load('en_core_web_sm', disable=['ner'])
nlp.remove_pipe('tagger')
nlp.remove_pipe('parser')


def tokenize(text: str) -> list:
    processed_text = nlp(text)
    processed_tokens = []
    for token in processed_text:
        if (not token.is_punct) and (not token.is_stop):
            processed_tokens.append(token.text.lower().strip())
    return processed_tokens