#TODO: do not run, not debugged.
import os
os.environ['TRANSFORMERS_CACHE'] = '/n/fs/nlp-jiatongy/iw22fall/base/.cache'
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline, set_seed
from evaluate import load

def eval_mauve():
    """
    
    """
