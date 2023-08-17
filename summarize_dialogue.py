from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import pandas as pd
import os
from os.path import join
import json
from episode import Episode
#from torchmetrics.text.rouge import ROUGEScore
from rouge_score import rouge_scorer


def split_text_by_lines(text):
    lines = text.split('\n')
    N = len(text.split())
    first_chunk_size = 0
    for i,l in enumerate(lines):
        first_chunk_size += len(l.split())
        if first_chunk_size > N/2:
            break
    first_chunk = '\n'.join(lines[:i+1])
    second_chunk = '\n'.join(lines[i+1:])
    return first_chunk, second_chunk

def summarize_scene(text):
    text = text.replace('@@ ','').replace(' ,',',')
    max_len = min(len(text.split()),50)
    min_len = max(10,max_len-20)
    print(len(text.split()))
    if len(text.split())>800:
        first_chunk, second_chunk = split_text_by_lines(text)
        return summarize_scene(first_chunk) + summarize_scene(second_chunk)
    try:
        return dpipe(text,min_length=min_len, max_length=max_len)[0]['summary_text']
    except IndexError:
        first_chunk, second_chunk = split_text_by_lines(text)
        return summarize_scene(first_chunk) + summarize_scene(second_chunk)

def get_rouges(pred,gt):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL','rougeLsum'], use_stemmer=True)
    raw_rscores = scorer.score(gt,pred)
    return {'r'+k[5:]+a: round(getattr(v,a),4) for k,v in raw_rscores.items() for a in ('precision','recall','fmeasure')}

def get_summ_of_summs(concatted_summs,gt_len):
    assert type(gt_len) in (int,float)
    if len(concatted_summs.split())>800:
        first_chunk, second_chunk = split_text_by_lines(concatted_summs)
        return get_summ_of_summs(first_chunk,gt_len/2) + get_summ_of_summs(second_chunk,gt_len/2)
    max_len = int(min(gt_len-50,250))
    min_len = int(max(90,max_len-40))
    summ_of_summs = pipe(concatted_summs,min_length=min_len, max_length=max_len)[0]['summary_text']
    return summ_of_summs

def harmonic_avg(args):
    return len(args)/sum([1/x for x in args])

if __name__ == '__main__':
    import openai
    openai.api_key = "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"

    dpipe = pipeline("summarization", model="kabita-choudhary/finetuned-bart-for-conversation-summary")
    pipe = pipeline("summarization", model="facebook/bart-large-cnn")

    all_our_bests = {}
    all_gpt_bests = {}
    for ep_fname in os.listdir('SummScreen/transcripts'):

        with open(join('SummScreen/transcripts',ep_fname)) as f:
            transcript_data = json.load(f)
        if not '[SCENE_BREAK]' in transcript_data['Transcript']: continue
        with open(join('SummScreen/summaries',ep_fname)) as f:
            summary_data = json.load(f)

        ep = Episode(transcript_data,summary_data)

        concatted_scene_summs = '\n'.join([summarize_scene(x) for x in ep.scenes])
        print('Concatted scene summaries:')
        print(len(concatted_scene_summs.split()))
        gt_len = sum([len(x.split()) for x in ep.summaries.values()])/len(ep.summaries)
        summ_of_summs = get_summ_of_summs(concatted_scene_summs,gt_len)
        gpt_summ = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k", messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": f"Please summarize the following TV show {ep.transcript}"},])['choices'][0]['message']['content']
        our_best = 0
        gpt_best = 0
        for summ_name, gt_summ in ep.summaries.items():
            print('\n'+summ_name)
            print('Summary of summaries:')
            our_scores = get_rouges(summ_of_summs,gt_summ)
            print(our_scores)
            our_avg = harmonic_avg([v for k,v in our_scores.items() if 'fmeasure' in k])
            if our_scores['r2fmeasure'] > our_best:
                our_best_scores = our_scores
                our_best = our_scores['r2fmeasure']
            print('GPT:')
            gpt_scores = get_rouges(gpt_summ,gt_summ)
            gpt_avg = harmonic_avg([v for k,v in gpt_scores.items() if 'fmeasure' in k])
            if gpt_scores['r2fmeasure'] > gpt_best:
                gpt_best_scores = gpt_scores
                gpt_best = gpt_scores['r2fmeasure']
            print(gpt_scores)

        print(f'\nBest ours: {our_best_scores}\nBest GPT: {gpt_best_scores}')
        all_our_bests[ep.title] = our_best_scores
        all_gpt_bests[ep.title] = gpt_best_scores
        if len(all_gpt_bests) == 2: break
    our_df = pd.DataFrame(all_our_bests).T
    gpt_df = pd.DataFrame(all_gpt_bests).T
    our_df.to_csv('our_rouge_scores.csv')
    gpt_df.to_csv('gpt_rouge_scores.csv')
