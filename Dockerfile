FROM nvcr.io/nvidia/pytorch:24.01-py3
WORKDIR /tmp
RUN mkdir ./data
COPY data/internvid-feats ./data/internvid-feats
COPY rag-caches ./rag-caches
COPY vrag.py ./vrag.py
COPY utils.py ./utils.py
COPY tvqa-splits.json ./tvqa-splits.json
#COPY data/postprocessed-video-captions ./data/postprocessed-video-captions
#COPY  ./data/postprocessed-video-captions
#RUN pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install -U langdetect
pip install py-rouge
pip install transformers
pip install datasets
pip install dl-utils385
pip install nltk
pip install natsort
pip install peft
pip install -U bitsandbytes
ENTRYPOINT ["python", "vrag.py", "--episode=2", "--cpu", "--model=llama3-8b "]
