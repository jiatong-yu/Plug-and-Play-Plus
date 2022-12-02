import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import GPT2LMHeadModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from data_util import get_dataloader

def load_model(logger, model_name="gpt2"):
    """
    Load transformers model, return tokenizer and model. 
    """
    free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
    max_memory = f'{free_in_GB-2}GB'
    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}
    logger.info(f"Loading {model_name} with {free_in_GB-2} GB memory and {n_gpus} GPU.")

    if "opt" in model_name:
        assert model_name in ["opt-350m", "opt-1.3b", "opt-2.7b", "opt-6.7b","opt-13b", "opt-30b", "opt-66b"]
        model_name = "facebook/"+model_name
    elif "galactica" in model_name:
        assert model_name in ["galactica-125m","galactica-1.3b","galactica-6.7b",'galactica-30b']
        model_name = "facebook/"+model_name
    elif "gpt2" in model_name:
        assert model_name in ["gpt2", "gpt2-medium","gpt2-large", "gpt2-xl"]
    
    else:
        raise NotImplementedError()



    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        max_memory=max_memory,
        offload_folder = "~/.cache/huggingface/.offload"
        )
    return model, tokenizer 
        



def generate(logger, 
            model, tokenizer, 
            input_tensors, batch_size = 64,
            method="beam", beam = 5,
            min_len = 0, max_len = 512,
            temperature = 0.8, lp = 1
            ):
    """
    Returns List[str] of model generation.
        input_tensors: Dict[str, List] from prepared_data (not dataloader).
    """
    assert method in ["beam", "greedy", "sampling"]

    dataloader = get_dataloader(input_tensors,batch_size=batch_size)
    generations = []
    
    logger.info("Running model generation...")
    for batch in tqdm(dataloader):
        input_ids = batch[0].cuda()
        attn_mask = batch[1].cuda()
        
        if method == "beam":
            outputs = model.generate(input_ids, attention_mask=attn_mask,
                                    max_length = max_len,
                                    min_length = min_len,
                                    temperature = temperature,
                                    num_beams = beam,
                                    length_penalty = lp,
                                    eos_token_id = tokenizer.eos_token_id,
                                    bos_token_id = tokenizer.bos_token_id,
                                    )
            generations += [_ for _ in tokenizer.batch_decode(outputs, skip_special_tokens=True)]
        
        elif method == "greedy":
            outputs = model.generate(input_ids, attention_mask=attn_mask,
                                    max_length = max_len,
                                    min_length = min_len,
                                    temperature = temperature,
                                    do_sample = False,
                                    length_penalty = lp,
                                    eos_token_id = tokenizer.eos_token_id,
                                    bos_token_id = tokenizer.bos_token_id,
                                    )
            generations += [_ for _ in tokenizer.batch_decode(outputs, skip_special_tokens=True)]
            

        elif method == "sampling":
            outputs = model.generate(input_ids, attention_mask=attn_mask,
                                    max_length = max_len,
                                    min_length = min_len,
                                    temperature = temperature,
                                    do_sample = True,
                                    length_penalty = lp,
                                    eos_token_id = tokenizer.eos_token_id,
                                    bos_token_id = tokenizer.bos_token_id,
                                    )
            generations += [_ for _ in tokenizer.batch_decode(outputs, skip_special_tokens=True)]
    
    logger.info(f"SHAPE OF GENERATIONS: {len(generations)}")
    return generations 