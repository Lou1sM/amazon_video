import re
import rouge


rouge_eval = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
                                                 max_n=2,
                                                 limit_length=False,
                                                 apply_avg=True,
                                                 apply_best=False,
                                                 alpha=0.5, # Default F1_score
                                                 stemming=False)

def rouge_preprocess(text):
    text = rouge.Rouge.REMOVE_CHAR_PATTERN.sub(' ', text.lower()).strip()
    tokens = rouge.Rouge.tokenize_text(rouge.Rouge.KEEP_CANNOT_IN_ONE_WORD.sub('_cannot_', text))
    rouge.Rouge.stem_tokens(tokens)
    preprocessed_text = rouge.Rouge.KEEP_CANNOT_IN_ONE_WORD_REVERSED.sub('cannot', ' '.join(tokens))
    return preprocessed_text

def nelly_rouge(pred,gt):
    if not isinstance(pred,list):
        pred = [pred]
    if not isinstance(gt,list):
        gt = [gt]
    pred_sums = [rouge_preprocess(pred) for pred in pred]
    gt_sums = [rouge_preprocess(g) for g in gt]

    scores = rouge_eval.get_scores(pred_sums, gt_sums)
    return scores

def extract_main_rouges(scores):
    rougel = scores['rouge-l']['f'] * 100
    rouge1 = scores['rouge-1']['f'] * 100
    rouge2 = scores['rouge-2']['f'] * 100
    return rouge1, rouge2, rougel

def rouge_from_multiple_refs(pred, references, return_full):
    best_r2 = -1
    for possible_gt in references:
        new_rouge = nelly_rouge(pred, possible_gt)
        if new_rouge['rouge-2']['f'] > best_r2:
            best_r2 = new_rouge['rouge-2']['f']
            best_rouge = new_rouge
    if best_r2 == 0:
        if not all([gt is None for gt in references]):
            print('rouge is zero')
    return best_rouge if return_full else extract_main_rouges(best_rouge)
def safe_decode(tokens, tokenizer):
     st = [[x for x in ts[:tokenizer.model_max_length] if x != -100] for ts in tokens]
     return tokenizer.batch_decode(st, skip_special_tokens=True, clean_up_tokenization_spaces=True)

def chunkify(text,max_chunk_size):
    if len(text.split())*4/3 < max_chunk_size:
        to_return = [text]
    else:
        first_chunk, second_chunk = split_text_by_sth(text)
        to_return = chunkify(first_chunk,max_chunk_size) + chunkify(second_chunk,max_chunk_size)
    if not all(len(x) <= max_chunk_size for sl in to_return for x in sl):
        breakpoint()
    return to_return

def split_text_by_sth(text):
    for sep in ('\n', '. ', ', ', ' '):
        if sep in text.strip():
            return split_text_by_sep(text.strip(),sep)
    return text[:len(text)//2], text[len(text)//2:]

def summ_short_scene(text):
    return ' '.join(convert_script_to_prose(line) for line in text.split('\n') if line!='')

def convert_script_to_prose(script_line):
    if maybe_speaker_name:=re.match(r'\w+: ', script_line):
        speaker_name = script_line[:maybe_speaker_name.span()[1]-2]
        speech = script_line[maybe_speaker_name.span()[1]:]
        return f'{speaker_name} said "{speech}"'
    elif stage_direction := re.match(r'(?<=\[ )[A-Z -]+(?= \])', script_line):
        return stage_direction
    else:
        return script_line

def split_text_by_sep(text,sep):
    lines = text.split(sep)
    N = len(text.split())
    first_chunk = ''
    for i,l in enumerate(lines):
        if abs(len((first_chunk+l).split()) - N/2) > abs(len(first_chunk.split())-N/2):
            break # get as close to halfway as possible
        if first_chunk=='':
            first_chunk = l
        else:
            first_chunk += sep+l
        if not text.startswith(first_chunk):
            breakpoint()
    second_chunk = text[len(first_chunk):]
    assert first_chunk+second_chunk == text
    return first_chunk, second_chunk

