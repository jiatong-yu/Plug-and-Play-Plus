import datasets 
from datasets import load_dataset, Dataset
import numpy as np
import pandas as pd 
import os 

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
    
    return data
    


