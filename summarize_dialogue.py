from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import json
from episode import Episode
from torchmetrics.text.rouge import ROUGEScore


tokenizer = AutoTokenizer.from_pretrained("kabita-choudhary/finetuned-bart-for-conversation-summary")
model = AutoModelForSeq2SeqLM.from_pretrained("kabita-choudhary/finetuned-bart-for-conversation-summary")


with open('SummScreen/tms_train.json') as f:
    data = json.load(f)

ep = Episode(data[0])

pipe = pipeline("summarization", model="kabita-choudhary/finetuned-bart-for-conversation-summary")
def summarize(text):
    print(text[:5])
    max_len = min(len(text.split()),50)
    min_len = max(10,max_len-20)
    return pipe(text,min_length=min_len, max_length=max_len)[0]
summarized_scene_dialogues = '\n'.join([summarize(x)['summary_text'] for x in ep.scenes])
rouge = ROUGEScore()
rouge(summarized_scene_dialogues,ep.summaries[0])
breakpoint()
