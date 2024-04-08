from .lm import LM
import openai
import time
import os
import numpy as np

class OpenAIModel(LM):
    def __init__(self, cache_file=None, key_path="api.key"):
        assert os.path.exists(key_path)
        self.key_path = key_path
        self.temp = 0.7
        self.save_interval = 100
        self.sent_cache = {}
        super().__init__(cache_file)

    def load_model(self):
        key_path = self.key_path
        assert os.path.exists(key_path), f"Please place your OpenAI APT Key in {key_path}."
        with open(key_path, 'r') as f:
            api_key = f.readline()
        self.api_key = api_key.strip()
        #self.model = self.model_name

    def _generate(self, prompt, max_output_tokens):
        to_send = [{'role':'user', 'content':prompt}]
        client = openai.OpenAI(api_key=self.api_key)
        prompt_end = prompt.split('\n\n')[-1]
        if (is_extration:=prompt_end.startswith('Please breakdown the following sentence into independent facts: ')):
            sent = prompt_end[64:]
            if sent in self.sent_cache:
                return self.sent_cache[sent]
        waittime = 2
        while True:
            try:
                response = client.chat.completions.create(
                  messages = to_send,
                  model='gpt-4-turbo-preview',
                  max_tokens=max_output_tokens,
                  temperature=self.temp,
                  top_p=0.9,
                  )
                break
            except openai.RateLimitError as e:
                print(f'{e}: waiting {waittime}')
                time.sleep(waittime)
                waittime = min(waittime*2, waittime+30)
        output = response.choices[0].message.content
        if is_extration:
            self.sent_cache[sent] = output
        return output
