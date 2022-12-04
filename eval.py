#TODO: do not run, not debugged.
print("launched eval pipeline.")
import os
import argparse
import numpy as np
from evaluate import load
import logging
from tqdm import tqdm

def calc_stats(logger, generations, labels):
    """
    Calculate the average number of words and sentences.
        generations in the format produced by run_baseline 
        labels in the format of prompt+text 
    """
    logger.info("calculating generation stats...")
    # word level 
    sum_ref = 0
    sum_res = 0
    for i in range(len(generations)):
        sum_ref += len(labels[i].split(" "))
        sum_res += len(generations[i].split(" "))
    logger.info(f"generation has avg. word {sum_res/len(generations)}")
    logger.info(f"reference has avg. word {sum_ref/len(labels)}")

    # word level 
    sum_ref = 0
    sum_res = 0
    for i in range(len(generations)):
        sum_ref += len(labels[i].split(". "))
        sum_res += len(generations[i].split(". "))
    logger.info(f"generation has avg. sent {sum_res/len(generations)}")
    logger.info(f"reference has avg. sent {sum_ref/len(labels)}")

    return 
    

def eval_mauve(logger,data_path):
    """
    Return mauve score of generations and references.
        data_path should contain a -ref.npy and a -res.npy as in run_baseline.py
    """
    logger.info(f"loading data from {data_path}")
    assert os.path.exists(data_path+"-res.npy")
    generations = np.load(data_path+"-res.npy")
    references = np.load(data_path+"-ref.npy",allow_pickle=True)

    assert len(generations) == len(references)
    
    if len(generations)<200: 
        logger.warning("To run mauve score, need at least 200 data points.")
    
    """
    TODO: if prompting method changed, this also need to be chagned.
    """
    

    mauve = load("mauve")

    labels=[]
    for data in references:
        if len(data["summary"]) > 1:
            summary_chunck = data["summary"].split(sep="\n")[0].split(sep=". ")[0]+". "
            prefix = "Generate long article based on title and summary. Title: "
            prompt = prefix+data["title"]+" Summary: "+ summary_chunck+"Generation: "
        
        else: 
            prefix = "Generate a long article based on title. Title: "
            prompt = prefix+data["title"]+" Generation: "
        
        label = prompt+data["text"]
        labels.append(label)
    
    calc_stats(logger, generations, labels)

    logger.info("calculating mauve scores...")
    mauve_score = mauve.compute(predictions=generations, references=labels)
    logger.info(f"mauve score: {mauve_score.mauve}")
    return mauve_score    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",type=str, default="gpt2-large")
    parser.add_argument("--res_dir",type=str, default="baseline1/cc_news-output/")
    parser.add_argument("--task",type=str, default="mauve")
    args = parser.parse_args()
    
    path = args.model + "-top-p"

    path = args.res_dir+path


    handlers = [logging.StreamHandler()]
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=handlers)
    logger = logging.getLogger(__name__)

    if args.task == "mauve":
        res = eval_mauve(logger, data_path=path)

    

    

