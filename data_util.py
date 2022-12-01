import datasets 
from datasets import load_dataset, Dataset
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import numpy as np
import pandas as pd 
import os 
from tqdm import tqdm 
import torch

def get_dataloader(inputs, batch_size=64):
    """
    Return dataloader. for each batch, input_ids = batch[0] and attn_mask = batch[1]
    """
    shape = inputs["input_ids"].shape 
    for v in inputs.values():
        assert v.shape == shape 
    
    dataset = TensorDataset(inputs["input_ids"], inputs["attn_mask"])
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    return dataloader
    

def load_data(logger, name="newsroom", split="validation", data_dir = "./data"):
    """
    Load data from newsroom dataset.
    
    Return
     data: Dataset object, with columns "title", "summary", "title"
    """
    if name == "newsroom":
        data = load_dataset(name, split=split, data_dir = data_dir).remove_columns(
            ['url', 'date', 'density_bin', 'coverage_bin', 'compression_bin', 'density', 'coverage', 'compression']
        )
        
    else: 
        raise NotImplementedError()
    
    logger.info(f"loaded {len(data)} from {name}.")
    return data
    


def prepare_data(logger,
                    tokenizer,data_list, 
                    title_prefix, content_prefix, 
                    summary_prefix=None, 
                    max_length=512):
    """
    Truncate each data to max_length and pad short samples, prepare data in tensor form.
        data_list: Dataset, keys {title, summary, text}
        title_prefix, content_prefix, summary_prefix: str 
    
    Prepare each data in the form title_prefix, title, (summary_prefix, summary), content_prefix.
    
    TODO: could optimize speed.

    Return 
        input_tensors: Dict[str, List] with keys {input_ids, attn_mask}
            input_ids: tokenizer output of prompt 
            attn_mask: mask out paddings
        
    """

    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id 

    logger.info("preparing data tensors...")
    
    input_ids_list = []
    attn_mask_list = []
    for data in tqdm(data_list):
        prompt = title_prefix+" "+data['title']+"\n"
        if summary_prefix is not None: 
            prompt = prompt + summary_prefix+" "+data['title']+"\n"
        
        prompt = prompt + content_prefix

        logger.info(prompt)
        
        tokens = tokenizer(prompt)["input_ids"][:max_length-2]

        tokens = [bos_token_id] + tokens 
        tokens = tokens + [eos_token_id]

        logger.info(f"length of tokens: {len(tokens)}")

        n_pad = max_length - len(tokens)
        input_ids = tokens + [0 for _ in range(n_pad)]
        attn_mask = [1 for _ in tokens] + [0 for _ in range(n_pad)]

        input_ids_list.append(input_ids)
        attn_mask_list.append(attn_mask)
    
    return {
        "input_ids": torch.LongTensor(input_ids_list),
        "attn_mask": torch.LongTensor(attn_mask_list)
    }