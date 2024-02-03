import argparse
from dl_utils.misc import check_dir
import string
import json
import numpy as np
import os
import logging

from tqdm import tqdm
from factscore.abstain_detection import is_response_abstained
from factscore.atomic_facts import AtomicFactGenerator
from factscore.clm import CLM
from factscore.npm import NPM
from factscore.openai_lm import OpenAIModel
from factscore.retrieval import DocDB, Retrieval

class FactScorer(object):
    def __init__(self, model_name, data_dir, model_dir, cache_dir, openai_key, cost_estimate, abstain_detection_type, batch_size=256):
        #assert model_name in ["retrieval+llama", "retrieval+llama+npm", "retrieval+ChatGPT", "npm", "retrieval+ChatGPT+npm"]
        self.model_name = model_name

        self.db = {}
        self.retrieval = {}
        self.npm = {}
        self.batch_size = batch_size # batch size for retrieval
        self.openai_key = openai_key
        self.abstain_detection_type = abstain_detection_type

        self.data_dir = data_dir
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        self.af_generator = None
        self.cost_estimate = cost_estimate

        if "llama" in model_name:
            if 'llama7B' in model_name:
                size = '7'
            elif 'llama13B' in model_name:
                size = '13'
            elif 'llama70B' in model_name:
                size = '70'
            self.lm = CLM(f"inst-llama-{size}B",
                          model_dir=os.path.join(model_dir, f"llama-{size}b"),
                          cache_file=os.path.join(cache_dir, f"llama-{size}b.pkl"))
        elif model_name == 'gpt-4-turbo-preview':
            self.lm = OpenAIModel(cache_file=os.path.join(cache_dir, "ChatGPT.pkl"),
                                  key_path=openai_key)
        else:
            self.lm = None

    def save_cache(self):
        if self.lm:
            self.lm.save_cache()
        if "npm" in self.model_name:
            for k, v in self.npm.items():
                v.save_cache()
        for k, v in self.retrieval.items():
            v.save_cache()

    def register_knowledge_source(self, name="enwiki-20230401", db_path=None, data_path=None):
        assert name not in self.retrieval, f"{name} already registered"
        if db_path is None:
            db_path = os.path.join(self.data_dir, f"{name}.db")

        if data_path is None:
            data_path = os.path.join(self.data_dir, f"{name}.jsonl")

        cache_path = os.path.join(self.cache_dir, f"retrieval-{name}.json")
        embed_cache_path = os.path.join(self.cache_dir, f"retrieval-{name}.pkl")

        self.db[name] = DocDB(db_path=db_path, data_path=data_path)
        self.retrieval[name] = Retrieval(self.db[name], cache_path, embed_cache_path, batch_size=self.batch_size)
        if "npm" in self.model_name:
            cache_path = os.path.join(self.cache_dir, f"bm25-{name}.json")
            embed_cache_path = os.path.join(self.cache_dir, f"bm25-{name}.pkl")
            self.npm[name] = NPM(Retrieval(self.db[name], cache_path, embed_cache_path, "bm25"),
                                 "npm-single",
                                 cache_file=os.path.join(self.cache_dir, f"npm-{name}.pkl"))

    def print_cost_estimates(self, total_words, task, model):
        total_tokens = total_words * 4.0 / 3
        if model == "davinci-003":
            rate = 0.02
        elif model == "gpt-3.5-turbo":
            rate = 0.002

        total_cost = total_tokens * rate / 1000

        logging.critical("Estimated OpenAI API cost for %s ($%.3f per 1000 tokens): $%.2f for %d words and %d tokens" % (task, rate, total_cost, total_words, total_tokens))

    def get_score(self, epname, topics, ref_summaries, generations=None, atomic_facts=None):
        assert type(topics)==list
        assert (generations is None) != (atomic_facts is None), 'use exactly one of gens/facts'
        if atomic_facts is not None:
            assert len(topics)==len(atomic_facts)
        else:
            if self.af_generator is None:
                self.af_generator = AtomicFactGenerator(key_path=self.openai_key,demon_dir=os.path.join(self.data_dir, "demos"),gpt3_cache_file=os.path.join(self.cache_dir, "InstructGPT.pkl"))

            # estimate the total cost of atomic fact generation
            total_words = 0
            for gen in generations:
                total_words += self.af_generator.run(gen, cost_estimate=self.cost_estimate)

            self.print_cost_estimates(total_words, task="atomic fact generation", model="davinci-003")
            #topics = tqdm(topics)
            atomic_facts = []
            for topic, gen in zip(topics, generations):
                # optionally, first detect if the response is abstained
                response_abstained = is_response_abstained(gen, self.abstain_detection_type)
                if response_abstained: # continue only when the response is not abstained
                    atomic_facts.append(None)
                    continue
                curr_afs, _ = self.af_generator.run(gen)
                curr_afs = [fact for _, facts in curr_afs for fact in facts]
                if len(curr_afs)==0:
                    atomic_facts.append(None)
                else:
                    atomic_facts.append(curr_afs)
                if len(atomic_facts) % 10 == 0:
                    self.af_generator.save_cache()

            assert len(atomic_facts)==len(topics)
            self.af_generator.save_cache()

        assert len(topics)==len(atomic_facts)
        respond_ratio = np.mean([facts is not None for facts in atomic_facts])

        #topics = tqdm(topics)
        scores = []
        decisions = []
        for topic, facts, ep_ref_summs in zip(topics, atomic_facts, ref_summaries):
            if facts is None:
                decisions.append(None)
            else:
                decision = self._get_score(epname, topic, facts, ep_ref_summs)
                if decision is None:
                    decisions.append(None)
                else:
                    score = np.mean([d["is_supported"] for d in decision])

                    decisions.append(decision)
                    scores.append(score)
                    if len(scores) % 10 == 0:
                        self.save_cache()

        self.save_cache()

        out = {"score": np.mean(scores),
               "respond_ratio": respond_ratio,
               "decisions": decisions,
               "num_facts_per_response": np.mean([len(d) for d in decisions if d is not None])}

        return out

    def _get_score(self, epname, topic, atomic_facts, summaries_dict, cost_estimate=None):
        decisions = []
        total_words = 0
        cache_dir = 'is_supported_factscore_caches/'
        print(f'\nScoring facts for {epname}\n')
        cache_path = os.path.join(cache_dir,f'{epname}-{self.model_name}.json')
        if (had_cache:=(check_dir(cache_dir) and os.path.exists(cache_path))):
            with open(cache_path) as f:
                cache = json.load(f)
        else:
            print('no is-supported cached found at', cache_path)
            cache = {}
        for i,atom in enumerate(atomic_facts):
            atom = atom.strip()
            definition = f'Answer the question about {topic} based on the given context.\n\n'
            for k,v in summaries_dict.items():
                definition += f'Title: {k}\nText: {v}\n\n'
            definition = ' '.join([x for x in definition.strip().split()][:3000])
            if not definition[-1] in string.punctuation:
                definition += "."
            prompt = f'{definition}\n\nInput: {atom.strip()} True or False?\nOutput:'

            if "ChatGPT" in self.model_name:
                # estimate the total cost of response generation
                n = len(prompt.split())
                self.print_cost_estimates(n, task="factscore evaluation", model="gpt-3.5-turbo")

            if atom in atomic_facts[:i]: # mark repeated facts as wrong
                if atom!='<MALFORMED SENTENCE>':
                    print('penalizing repeated fact')
                is_supported = False
            elif atom =='<MALFORMED SENTENCE>':
                is_supported = False
            elif 'airs on' in atom.lower() or 'season finale' in atom.lower():
                is_supported = False
            elif 'click' in atom.lower() or 'link' in atom.lower():
                is_supported = False
            elif 'Samaritans' in atom:
                is_supported = False
            elif '.com' in atom:
                is_supported = False
            elif atom in cache:
                is_supported = cache[atom]
            else:
                if had_cache:
                    print('atom:', atom, 'not in cache at', cache_path)
                output = self.lm.generate(prompt)

                if not isinstance(output, str) and type(output[1])==np.ndarray:# when logits are available
                    logits = np.array(output[1])
                    assert logits.shape[0] in [32000, 32001]
                    true_score = logits[5852]
                    false_score = logits[7700]
                    is_supported = true_score > false_score
                else:# when logits are unavailable
                    if isinstance(output, list):
                        assert len(output)==1
                        output = output[0]
                    generated_answer = output.lower()
                    if "true" in generated_answer or "false" in generated_answer:
                        if "true" in generated_answer and "false" not in generated_answer:
                            is_supported = True
                        elif "false" in generated_answer and "true" not in generated_answer:
                            is_supported = False
                        else:
                            is_supported = generated_answer.index("true") > generated_answer.index("false")
                    else:
                        is_supported = all([kw not in generated_answer.lower().translate(str.maketrans("", "", string.punctuation)).split() for kw in ["not", "cannot", "unknown", "information"]])
                cache[atom] = bool(is_supported)

            print(atom, is_supported)
            decisions.append({"atom": atom, "is_supported": is_supported})

        if had_cache:
            with open(cache_path) as f:
                orig_cache = json.load(f)
            for k,v in orig_cache.items():
                assert cache[k] == v

        with open(cache_path, 'w') as f:
            #print('saving cache to', cache_path)
            json.dump(cache, f)
        if cost_estimate:
            return total_words
        else:
            return decisions
