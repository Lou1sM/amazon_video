from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import json
from episode import Episode
from torchmetrics.text.rouge import ROUGEScore


#tokenizer = AutoTokenizer.from_pretrained("kabita-choudhary/finetuned-bart-for-conversation-summary")
#model = AutoModelForSeq2SeqLM.from_pretrained("kabita-choudhary/finetuned-bart-for-conversation-summary")


dpipe = pipeline("summarization", model="kabita-choudhary/finetuned-bart-for-conversation-summary")
pipe = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_scene(text):
    max_len = min(len(text.split()),50)
    min_len = max(10,max_len-20)
    return dpipe(text,min_length=min_len, max_length=max_len)[0]['summary_text']

def get_rouges(pred,gt):
    rouge = ROUGEScore()
    raw_rscores =rouge(pred,gt)
    return {'r'+k[5:]: round(v.item(),4) for k,v in raw_rscores.items()}

def get_summ_of_summs(concatted_summs,gt):
    max_len = min(len(gt.split())-50,250)
    min_len = max(90,max_len-40)
    summ_of_summs = pipe(concatted_summs,min_length=min_len, max_length=max_len)[0]['summary_text']
    return summ_of_summs


if __name__ == '__main__':
    with open('SummScreen/tms_train.json') as f:
        data = json.load(f)

    for ep_dict in data[:5]:
        ep = Episode(ep_dict)

        concatted_scene_summs = '\n'.join([summarize_scene(x) for x in ep.scenes])
        print('Concatted scene summaries:')
        print(get_rouges(concatted_scene_summs,ep.summaries[0]))

        summ_of_summs = get_summ_of_summs(concatted_scene_summs,ep.summaries[0])
        print('Summary of summaries:')
        print(get_rouges(summ_of_summs,ep.summaries[0]))
