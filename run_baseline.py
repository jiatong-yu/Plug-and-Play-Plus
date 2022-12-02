print("running baseline evaluations.")
import os 
import torch
import logging
import argparse
import random

from data_util import load_data, prepare_data, get_dataloader
from model_util import load_model, generate

def main(logger, args):

    raw_data = load_data(logger, split=args.split)
    DEBUG = True 
    if DEBUG:
        data = raw_data.select([random.randint(0,len(raw_data)-1) for i in range(args.n_sample)])
    else: 
        data = raw_data
    model, tokenizer = load_model(logger, args.model)

    
    if args.include_summary:
        summary_prefix = "Summary:"
    else: 
        summary_prefix = None 
    input_tensors = prepare_data(logger, tokenizer,
                                 data,
                                 title_prefix= "Title:",
                                 content_prefix= "Content:", 
                                 summary_prefix= summary_prefix,
                                 max_length=args.input_max_len
                                 )

    generations = generate(logger, model, tokenizer,
                            input_tensors, batch_size = args.batch_size,
                            method=args.method, beam=args.num_beam,
                            min_len=args.generate_min_len,
                            max_len=args.generate_max_len)

    if not args.disgard_gen:
        out_dir = args.out_dir
        assert os.path.exists(out_dir)
        
        # base = out_dir + "/" + args.model+"-"+args.method+"-result-1.txt"
        # reference = out_dir + "/" + args.model+"-reference-1.txt"

        base = args.out_dir+"/{}-{}".format(args.model, args.method)
        reference = args.out_dir+"/{}-{}".format(args.model, args.method)

        if args.include_summary:
            base = base+"-sum"
            reference = reference + "-sum"
        
        base = base+"-res.txt"
        reference = reference+"-ref.txt"
        
        f = open(base,"w")
        r = open(reference,"w")
        for i in range(len(data)):
            f.write(generations[i]+"\n\n\n")
            r.write(data[i]+"\n\n\n")
            # print(generations[i])



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model",type=str, default="gpt2")
    parser.add_argument("--method",type=str, default="sampling")
    parser.add_argument("--split",type=str, default="test")
    parser.add_argument("--num_beam",type=int, default=5)
    parser.add_argument("--generate_min_len",type=int,default=0)
    parser.add_argument("--input_max_len",type=int,default=126)
    parser.add_argument("--generate_max_len",type=int,default=512)
    parser.add_argument("--batch_size",type=int, default=10)
    parser.add_argument("--include_summary",default=False, action="store_true")

    parser.add_argument("--n_sample",type=int,default=50)
    parser.add_argument("--disgard_gen",default=False,action="store_true")
    parser.add_argument("--out_dir",type=str,default="output")

    
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