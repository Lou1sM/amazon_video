from .lm import LM
import openai
import sys
import time
import os
import numpy as np
import logging

class OpenAIModel(LM):
    def __init__(self, cache_file=None, key_path="api.key"):
        self.key_path = key_path
        self.temp = 0.7
        self.save_interval = 100
        super().__init__(cache_file)

    def load_model(self):
        key_path = self.key_path
        assert os.path.exists(key_path), f"Please place your OpenAI APT Key in {key_path}."
        with open(key_path, 'r') as f:
            api_key = f.readline()
        self.api_key = api_key.strip()
        #self.model = self.model_name

    def _generate(self, prompt, max_output_length=128):
        to_send = [{'role':'user', 'content':prompt}]
        client = openai.OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
              messages = to_send,
              model='gpt-4-turbo-preview',
              max_tokens=512,
              temperature=self.temp,
              top_p=0.9,
              )
        output = response.choices[0].message.content
        print(output)
        return output
