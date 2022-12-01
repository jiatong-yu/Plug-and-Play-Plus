print("running baseline evaluations.")
import os 
import torch
import logging
import argparse

from data_util import load_data, prepare_data, get_dataloader
from model_util import load_model, generate

def main(logger, args):

    raw_data = load_data(logger, split=args.split)
    model, tokenizer = load_model(logger, args.model)

    DEBUG = True 
    if DEBUG:
        data = raw_data.select([i for i in range(50)])
    else: 
        data = raw_data
    
    input_tensors = prepare_data(logger, tokenizer,
                                 data,
                                 title_prefix= "Title:",
                                 content_prefix= "Content:", 
                                 summary_prefix="Summary:",
                                 max_length=args.input_max_len
                                 )

    generations = generate(logger, model, tokenizer,
                            input_tensors, batch_size = args.batch_size,
                            method=args.method, beam=args.num_beam,
                            min_len=args.generate_min_len,
                            max_len=args.generate_max_len)

    if DEBUG:
        for gen in generations:
            print(gen)
            print("======================")
    return 



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model",type=str, default="gpt2")
    parser.add_argument("--method",type=str, default="beam")
    parser.add_argument("--split",type=str, default="test")
    parser.add_argument("--num_beam",type=int, default=5)
    parser.add_argument("--generate_min_len",type=int,default=0)
    parser.add_argument("--input_max_len",type=int,default=126)
    parser.add_argument("--generate_max_len",type=int,default=512)
    parser.add_argument("--batch_size",type=int, default=64)


    
    args = parser.parse_args()

    assert args.split in ["validation","train","test"]
    
    handlers = [logging.StreamHandler()]
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=handlers)
    logger = logging.getLogger(__name__)
    logger.info(args)
    main(logger, args)