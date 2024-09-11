FROM python:latest
WORKDIR /tmp
COPY utils.py ./
COPY summarize_dialogue.py ./
COPY episode.py ./
RUN mkdir ./data
COPY data/transcripts ./data/transcripts
COPY data/postprocessed-video-captions ./data/postprocessed-video-captions
#RUN pip install --no-cache-dir --upgrade pip && \
    #pip install --no-cache-dir --upgrage transformers dl-utils385
RUN pip install py-rouge
RUN pip install transformers
RUN pip install dl-utils385
RUN pip install nltk
RUN pip install torch --index-url https://download.pytorch.org/whl/cu124
RUN pip install natsort
RUN pip install peft
#ENTRYPOINT ["python", "summarize_dialogue.py", "--device=cpu"]
