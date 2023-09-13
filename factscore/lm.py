import pickle
import os
import time


class LM(object):
    def __init__(self, cache_file):
        self.cache_file = cache_file
        self.cache_dict = self.load_cache()
        self.model = None
        self.add_n = 0

    def load_model(self):
        # load the model and put it as self.model
        raise NotImplementedError()

    def generate(self, pred=None, path=None, sample_idx=0, max_output_length=128):
        assert (pred is None) != (path is None)
        if pred is None:
            with open(path) as f:
                pred = f.read()
        if self.model is None:
            self.load_model()

        max_output_length = 1 if pred.endswith(" True or False?\nAnswer:") else max_output_length
        pred = pred.strip() # it's important not to end with a whitespace
        return self._generate(pred, max_output_length=max_output_length)

    def save_cache(self):
        if self.add_n == 0:
            return

        # load the latest cache first, since if there were other processes running in parallel, cache might have been updated
        for k, v in self.load_cache().items():
            self.cache_dict[k] = v

        with open(self.cache_file, "wb") as f:
            pickle.dump(self.cache_dict, f)

    def load_cache(self, allow_retry=True):
        if self.cache_file is not None and os.path.exists(self.cache_file):
            while True:
                try:
                    with open(self.cache_file, "rb") as f:
                        cache = pickle.load(f)
                    break
                except Exception:
                    if not allow_retry:
                        assert False
                    print ("Pickle Error: Retry in 5sec...")
                    time.sleep(5)
        else:
            cache = {}
        return cache
