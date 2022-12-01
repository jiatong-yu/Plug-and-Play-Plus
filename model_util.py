import numpy as np
import os
import torch
import torch.nn.functional as F
import logging 
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import GPT2LMHeadModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(logger, model_name="gpt2"):
    """
    Load transformers model, return tokenizer and model. 
    """
    free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
    max_memory = f'{free_in_GB-2}GB'
    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}
    logger.info(f"Loading {model_name} with {free_in_GB-2} GB memory and {n_gpus} GPU.")

    # Load gpt2 model of different sizes 
    if "gpt2" in model_name:
        assert model_name in ["gpt2", "gpt2-medium","gpt2-large", "gpt2-xl"]
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        torch.cuda.empty_cache()
        model = GPT2LMHeadModel.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            max_memory=max_memory,
            offload_folder = "~/.cache/huggingface/.offload"
            )
        return model, tokenizer

    # load opt models 
    if "opt" in model_name:
        assert model_name in ["opt-350m", "opt-1.3b", "opt-2.7b", "opt-6.7b","opt-13b", "opt-30b", "opt-66b"]
        model_name = "facebook/"+model_name
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map='auto',
            torch_dtype=torch.float16,
            max_memory=max_memory,
            offload_folder = "~/.cache/huggingface/.offload"
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        return model, tokenizer 
    
    if "galactica" in model_name:
        assert model_name in ["galactica-125m","galactica-1.3b","galactica-6.7b",'galactica-30b']
        model_name = "facebook/"+model_name
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map='auto',
            torch_dtype=torch.float16,
            max_memory=max_memory,
            offload_folder = "~/.cache/huggingface/.offload"
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        return model, tokenizer 
        

    raise NotImplementedError("model input not supported.")

