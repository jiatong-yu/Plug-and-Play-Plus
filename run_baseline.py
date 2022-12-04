print("running baseline evaluations.")
import os 
import torch
import logging
import argparse
import random
import numpy as np
from data_util import load_data, prepare_data, get_dataloader
from model_util import load_model, generate

def main(logger, args):

    raw_data = load_data(logger, name=args.data_task, split=args.split)
    DEBUG = True 
    if DEBUG:
        data = raw_data.select([random.randint(0,len(raw_data)-1) for i in range(args.n_sample)])
    else: 
        data = raw_data

    model, tokenizer = load_model(logger, args.model)
    
    input_tensors = prepare_data(logger, tokenizer, data)


    generations = generate(logger, model, tokenizer,
                            input_tensors, batch_size = args.batch_size,
                            method=args.method,
                            min_len=args.generate_min_len,
                            max_len = args.generate_max_len,
                            )

    if not args.disgard_gen:
        out_dir = "baseline1/"+args.data_task+"-output"
        assert os.path.exists(out_dir)
        

        base = out_dir+"/{}-{}".format(args.model, args.method)
        reference = out_dir+"/{}-{}".format(args.model, args.method)

        
        base = base+"-res.npy"
        reference = reference+"-ref.npy"
        f = open(base, "wb")
        r = open(reference,"wb")
        base = base+"-res.npy"
        reference = reference+"-ref.npy"
        np.save(f, generations)
        np.save(r, data)
        
        """
        Uncomment to save in txt format.
        """
        # base = base+"-res.txt"
        # reference = reference+"-ref.txt"
        # f = open(base,"w")
        # r = open(reference,"w")
        # for i in range(len(generations)):
        #     f.write(generations[i]+"\n")
        #     f.write(50*"-"+"\n")
        #     r.write(data[i]['title']+":"+data["summary"].split(sep="\n")[0].split(sep=". ")[0]+"\n")
        #     r.write(data[i]["text"]+"\n")
        #     r.write(50*"-"+"\n")



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model",type=str, default="gpt2")
    parser.add_argument("--method",type=str, default="top-p")
    parser.add_argument("--split",type=str, default="test")
    parser.add_argument("--generate_min_len",type=int,default=256)
    parser.add_argument("--generate_max_len",type=int,default=640)
    parser.add_argument("--batch_size",type=int, default=10)
    parser.add_argument("--data_task",type=str, default="cc_news")

    parser.add_argument("--n_sample",type=int,default=50)
    parser.add_argument("--disgard_gen",default=False,action="store_true")

    
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