#!/usr/bin/env python3
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import GenerationConfig
from qwen_generation_utils import make_context, decode_tokens, get_stop_words_ids
#import sys
#sys.path.append('qwen_generation_utils')


def chat(model_path):
    device = "cuda:0"
    #tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side='left')
    # from docx readme
    tokenizer = AutoTokenizer.from_pretrained(
        #'./',
        model_path,
        pad_token='<|extra_0|>',
        eos_token='<|endoftext|>',
        padding_side='left',
        trust_remote_code=True
    )


    #model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda:0", trust_remote_code=True).to(device)

    # i fix this shit my self!!!!!!!! add that device_map to it
    #model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda:0", trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        #'./',
        model_path,
        pad_token_id=tokenizer.pad_token_id,
        device_map="cuda:0",
        trust_remote_code=True
    ).eval()

    # comment out currently
    #model.generation_config = GenerationConfig.from_pretrained(model_path, pad_token_id=tokenizer.pad_token_id)


    # Check if the tokenizer has an EOS token set; if not, manually set it.
    if not tokenizer.eos_token:
       tokenizer.eos_token = "<|endoftext|>"  # Ensure this matches your model's training




    
    # test cases

    response, history = model.chat(tokenizer, "你好", history=None)
    print(response)



    model.generation_config = GenerationConfig.from_pretrained(model_path, pad_token_id=tokenizer.pad_token_id)
    
    all_raw_text = ["我想听你说爱我。", "今天我想吃点啥，甜甜的，推荐下", "我马上迟到了，怎么做才能不迟到"]
    batch_raw_text = []
    for q in all_raw_text:
        raw_text, _ = make_context(
            tokenizer,
            q,
            system="You are a helpful assistant.",
            max_window_size=model.generation_config.max_window_size,
            chat_format=model.generation_config.chat_format,
        )
        batch_raw_text.append(raw_text)
    
    batch_input_ids = tokenizer(batch_raw_text, padding='longest')
    batch_input_ids = torch.LongTensor(batch_input_ids['input_ids']).to(model.device)
    batch_out_ids = model.generate(
        batch_input_ids,
        return_dict_in_generate=False,
        generation_config=model.generation_config
    )
    padding_lens = [batch_input_ids[i].eq(tokenizer.pad_token_id).sum().item() for i in range(batch_input_ids.size(0))]
    
    batch_response = [
        decode_tokens(
            batch_out_ids[i][padding_lens[i]:],
            tokenizer,
            raw_text_len=len(batch_raw_text[i]),
            context_length=(batch_input_ids[i].size(0)-padding_lens[i]),
            chat_format="chatml",
            verbose=False,
            errors='replace'
        ) for i in range(len(all_raw_text))
    ]
    print(batch_response)
    
    response, _ = model.chat(tokenizer, "我想听你说爱我。", history=None)
    print(response)
    
    response, _ = model.chat(tokenizer, "今天我想吃点啥，甜甜的，推荐下", history=None)
    print(response)
    
    response, _ = model.chat(tokenizer, "我马上迟到了，怎么做才能不迟到", history=None)
    print(response)



    '''
    print("========== Chat mode starts ==========")
    while True:
        input_text = input("You: ")
        if input_text.lower() == 'quit':
            break
        
        # Encode the input text
        input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt').to(device)

        # Generate a response
        with torch.no_grad():
            #output_ids = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
            #output_ids = model.generate(input_ids, pad_token_id=tokenizer.eos_token_id)
            output_ids = model.generate(input_ids,
                            temperature=0.9,  # Adjust for randomness
                            top_k=50,  # Keep top 50 tokens for sampling
                            top_p=0.95,  # Use nucleus sampling
                            no_repeat_ngram_size=2,  # Avoid repeating n-grams
                            num_return_sequences=1,  # Number of sequences to return
                            pad_token_id=tokenizer.eos_token_id)  # Ensure padding is correct

        
        # Decode and print the response
        response = tokenizer.decode(output_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
        print("Model:", response)
        '''

if __name__ == "__main__":
    model_path = "/home/eawlot3000/cherish/Qwen/finetune/output_qwen/checkpoint-2000"  # Update this to your model's path
    chat(model_path)
