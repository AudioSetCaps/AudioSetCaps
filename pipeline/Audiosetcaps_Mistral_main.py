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
import sys
import string
import pandas as pd
import numpy as np
import requests

import laion_clap

def load_CLAP(pretrained_path):
    CLAP_model = laion_clap.CLAP_Module(enable_fusion=False, amodel= 'HTSAT-base')
    CLAP_model.load_ckpt(pretrained_path)
    return CLAP_model

def cal_cos_sim(model, audio_path, input_text_data_list):
    cos_sim_list = []
    ### Directly get audio embeddings from audio files
    audio_file = [audio_path]
    audio_emb = model.get_audio_embedding_from_filelist(x = audio_file, use_tensor=False)
    audio_emb = audio_emb.reshape(-1)

    text_embed_list = model.get_text_embedding(input_text_data_list)
    for text_emb in text_embed_list:
        cos_sim = audio_emb.dot(text_emb)/(np.linalg.norm(audio_emb)*np.linalg.norm(text_emb))
        cos_sim_list.append(cos_sim)
    
    return cos_sim_list  

def pad_str(str_list, pad_mode):
    if pad_mode == "min":
        min_len = min(len(s) for s in str_list)
        new_str_list = [s[:min_len] for s in str_list]
        
    elif pad_mode=="max":
        max_length = max(len(s) for s in str_list)
        new_str_list = [s + s * ((max_length - len(s)) // len(s)) for s in str_list]
    return new_str_list   

def pad_as_AC_str(str_list):
    AC_str = str_list[-1]
    GPT_str = int(np.ceil(len(str_list[0])/len(AC_str)))*AC_str
    Mistral_str = int(np.ceil(len(str_list[1])/len(AC_str)))*AC_str
    str_list[0] = GPT_str
    str_list[1] = Mistral_str
    
    return str_list

def pad_as_lb_str(str_list):
    lb_str = str_list[-1]+'.'
    GPT_str = int(np.ceil(len(str_list[0])/len(lb_str)))*lb_str
    Mistral_str = int(np.ceil(len(str_list[1])/len(lb_str)))*lb_str
    AC_str = int(np.ceil(len(str_list[2])/len(lb_str)))*lb_str
    str_list[0] = GPT_str
    str_list[1] = Mistral_str
    str_list[2] = AC_str
    
    return str_list      
     

def remove_punctuation(input_string):

    translator = str.maketrans(string.punctuation, " "*len(string.punctuation))
    result_string = input_string.translate(translator)
    
    return result_string

def LLMs_main(config):

    print('######## LLMs Captioning! ########')    
    
    as_csv = pd.read_csv(config.as_csv_path, sep=',')    
    
    with open(config.default_prompt_path, 'r') as log_f:
        log_lines = log_f.read()
    default_prompt = log_lines
    
    speech_keyword = ["no speech", "does not", "is not"]
    music_keyword = ["no music", "does not", "is not"]
    
    CLAP_model = load_CLAP(config.pretrained_path)
    print('######## Loading CLAP model! ########') 
    
    filename_list = []
    label_list = []
    CLAP_label_score = []
    CLAP_Mistral_score = []
    
    for index, row in as_csv.iterrows(): 
        
        filename = row['filename']
        wav_path = os.path.join(config.root_audio_path, filename)   
        label = row['label']
        
        qwen_caption_save_path = os.path.join(config.qwen_caption_path, filename[:-4]+'.txt')
        os.makedirs(config.mis_caption_path, exist_ok=True)
        mis_caption_save_path = os.path.join(config.mis_caption_path, filename[:-4]+'.txt')
        
        if os.path.exists(qwen_caption_save_path) and not os.path.exists(mis_caption_save_path): #
                print(index, filename)
            # try:
                for i in range(config.mistral_try_num):
                    best_cap_score = -10
                    best_cap = ''
                    
                    with open(qwen_caption_save_path, 'r') as c_f:   
                        qwen_caption = c_f.readlines()
    
                    speech_caption = qwen_caption[1]
                    music_caption = qwen_caption[2]
                    
                    if any(keyword in speech_caption for keyword in speech_keyword):
                        speech_caption = ""
                    if any(keyword in music_caption for keyword in music_keyword):
                        music_caption = ""
                    qwen_prompt = 'Details:\n1 Crowd-sourced workers:{'+'{}{}{}'.format(qwen_caption[0], speech_caption, music_caption)+'}'
                    qwen_prompt = qwen_prompt+'\n'+'2 The ground truth labels:{'+'{}'.format(label)+'}'+'\n'
                    qwen_prompt = qwen_prompt+"My instructions:\n\
                        Do not mention the specific content of speech! Do not mention the specific content of speech!\
                        Do not output the ground truth labels in the caption! Do not output the ground truth labels in the caption!\
                        \nOutput your caption (Within 50 words):"
    
                    request_data = {
                        "model": "mistral",
                        "messages": [{"role": "system", "content": default_prompt}, 
                                      {"role": "user", "content": qwen_prompt}],
                        "stream": False
                    }
                    
                    url = "http://localhost:11434/api/chat"
                    response = requests.post(url, json=request_data)
                    
                    response_data = response.json()
                    content = response_data.get("message", {}).get("content")
                    
                    ### delete the punctuations
                    input_content = remove_punctuation(content)
                    input_label = remove_punctuation(label)
                    
                    ### pad to caption length must be the format [your_text, label] !!!!
                    input_text = [input_content, input_label]
                    input_text = pad_str(input_text, "max")
                    
                    # print(input_text)
                    
                    [mis_cos_sim, lb_cos_sim] = cal_cos_sim(CLAP_model, wav_path, input_text)
                    
                    if mis_cos_sim>=lb_cos_sim:
                        best_cap_score = mis_cos_sim
                        best_cap = content
                        break
                    elif i==0 or mis_cos_sim>=best_cap_score:
                        best_cap_score = mis_cos_sim
                        best_cap = content
                
                with open(mis_caption_save_path, 'w') as out_f:
                    out_f.write(best_cap)
                
                filename_list.append(filename)
                label_list.append(label)
                CLAP_Mistral_score.append(best_cap_score)
                CLAP_label_score.append(lb_cos_sim)
                
                CLAP_score_df = pd.DataFrame({'filename':filename_list, 'label':label_list, "Mistral_score":CLAP_Mistral_score, 'label_score':CLAP_label_score})
                CLAP_score_df.to_csv(config.CLAP_score_csv_path, index=False) 
            
            # except:
            #     continue
            
            
if __name__ == "__main__":
    
    LLMs_main(config)
