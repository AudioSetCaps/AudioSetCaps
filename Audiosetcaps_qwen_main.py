# -*- coding: utf-8 -*-
# AudioSetCaps: Enriched Audio Captioning Dataset Generation Using Large Audio Language Models
# Jisheng Bai, Haohe Liu
# Northwestern Polytechnical University, Xi'an Lianfeng Acoustic Technologies Co., Ltd.
# CVSSP, University of Surrey

import config
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import pandas as pd

import torch
from modelscope import (snapshot_download, AutoModelForCausalLM, AutoTokenizer, GenerationConfig)

def load_Qwen_audio(model_id, revision, cache_dir):

    model_dir = snapshot_download(model_id, revision=revision, cache_dir=cache_dir)
    torch.manual_seed(1234)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if not hasattr(tokenizer, 'model_dir'):
        tokenizer.model_dir = model_dir

    qwen_model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, fp16=True).eval()
    qwen_model = torch.compile(qwen_model)
    
    return tokenizer, qwen_model

def qwen_audio_main(config):
    print(config.as_csv_path)
    print('######## Qwen-audio Captioning! ########')
    tokenizer, model = load_Qwen_audio(config.model_id, config.revision, config.cache_dir)

    as_csv = pd.read_csv(config.as_csv_path, sep=',')

    for index, row in as_csv.iterrows():
        
        filename = row['filename']
        wav_path = os.path.join(config.root_audio_path, filename)     
        if os.path.exists(wav_path):
            caption_save_path = os.path.join(config.qwen_caption_path, filename[:-4]+'.txt')
            if not os.path.exists(caption_save_path):
                print(index, filename)
                try:
                    for i in range(config.qwen_try_num):
                        response = ''
                        query = tokenizer.from_list_format([
                            {'audio': wav_path},
                            {'text': 'Describe this audio according to the sounds in it within 50 words.'},
                        ])
                        re1, history = model.chat(tokenizer, query=query, history=None)
                        response = response+'Description: '+re1+'\n'
                        query = tokenizer.from_list_format([
                            {'text': 'Based on the QAs, give some information about the speech, such as the emotion of the speaker, the gender of the speaker, and the spoken language, only if speech is present in this audio.'},
                        ])
                        re2, history = model.chat(tokenizer, query=query, history=history)
                        response = response+'Speech: '+re2+'\n'
                        query = tokenizer.from_list_format([
                            {'text': 'Based on the QAs, give some information about the music, such as music genre and music instruments, only if music is present in this audio.'},
                        ])
                        re3, history = model.chat(tokenizer, query=query, history=history)
                        response = response+'Music: '+re3
                        # print(response)
                        os.makedirs(config.qwen_caption_path, exist_ok=True)
                        with open(caption_save_path,'w') as f:
                            f.write(response)

                        response_str_len = len(response)
                        if response_str_len>=100 and response_str_len<=500:
                            break
                
                except:
                    continue
            
if __name__ == "__main__":
    
    qwen_audio_main(config)
