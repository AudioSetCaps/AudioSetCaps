#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# AudioSetCaps: An Enriched Audio-Caption Dataset using Automated Generation Pipeline with Large Audio and Language Models
# Jisheng Bai, Haohe Liu
# Northwestern Polytechnical University, Xi'an Lianfeng Acoustic Technologies Co., Ltd.
# CVSSP, University of Surrey

import librosa
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from models.ase_model import ASE
import torch.nn.functional as F
import os
import math


### config load
config = yaml.load(open("./inference.yaml", "r"), Loader=yaml.FullLoader)

### model ckpt load
device = "cuda"
model = ASE(config)
model.to(device)
ckpt_path = config['ckpt_path']
cp = torch.load(ckpt_path)
model.load_state_dict(cp['model'], strict=False)
model.eval()
print("Model weights loaded from {}".format(ckpt_path))

### ESC-50 ##########
print("### ESC50 ZSC ###")
batch_size = 32
df = pd.read_csv('~/esc50.csv')
class_to_idx = {}
sorted_df = df.sort_values(by=['target'])
classes = ["The clip showcases "+x.replace('_', ' ') for x in sorted_df['category'].unique()]

print(classes[:10])

pre_path = '~/ESC-50-master/audio/'

batch_size = 32
audio_length = 32000 * 10

with torch.no_grad():
    text_embeds = model.encode_text(classes)
    fold_acc = []
    for fold in range(1, 6):
        fold_df = sorted_df[sorted_df['fold'] == fold]
        y_preds, y_labels = [], []
        
        for i in tqdm(range(0, len(fold_df), batch_size)):
            batch_df = fold_df.iloc[i:i+batch_size]
            
            # Load and preprocess batch
            batch_audio = []
            for file_path in batch_df["filename"]:
                audio, _ = librosa.load(os.path.join(pre_path, file_path), sr=32000, mono=True)
                audio = torch.tensor(audio)
                if audio.shape[-1] < audio_length:
                    repeat_times = math.ceil(audio_length / audio.shape[-1])
                    audio = audio.repeat(repeat_times)
                    audio = audio[:audio_length]
                elif audio.shape[-1] > audio_length:
                    audio = audio[:audio_length]
                batch_audio.append(audio)
            
            batch_audio = torch.stack(batch_audio).to(device)
            batch_targets = torch.tensor(batch_df["target"].values)
            one_hot_targets = F.one_hot(batch_targets, num_classes=len(classes)).float()
            
            # Process batch
            audio_emb = model.encode_audio(batch_audio)
            similarity = audio_emb @ text_embeds.t()
            y_pred = F.softmax(similarity, dim=1)
            
            y_preds.append(y_pred.cpu())
            y_labels.append(one_hot_targets.cpu())
        
        y_labels = torch.cat(y_labels, dim=0).numpy()
        y_preds = torch.cat(y_preds, dim=0).numpy()
        
        acc = accuracy_score(np.argmax(y_labels, axis=1), np.argmax(y_preds, axis=1))
        print(f'Fold {fold} Accuracy {acc}')
        fold_acc.append(acc)

print(f'ESC50 Accuracy {np.mean(fold_acc)}')

### Urbansound8k #########################################
# print("### Urbansound8k ZSC ###")
# df = pd.read_csv('data/UrbanSound8K/metadata/UrbanSound8K.csv')
# sorted_df = df.sort_values(by=['classID'])
# classes = ["{} can be heard".format(x.replace('_', ' ')) for x in sorted_df['class'].unique()]

# print(classes[:10])
# pre_path = '/ssd2/BJS_data/UrbanSound8K/audio/'

# batch_size = 32
# audio_length = 32000 * 10

# with torch.no_grad():
#     text_embeds = model.encode_text(classes)
#     fold_acc = []
#     for fold in range(1, 11):
#         fold_df = sorted_df[sorted_df['fold'] == fold]
#         y_preds, y_labels = [], []
        
#         for i in tqdm(range(0, len(fold_df), batch_size)):
#             batch_df = fold_df.iloc[i:i+batch_size]
            
#             # Load and preprocess batch
#             batch_audio = []
#             batch_targets = []
#             for file_path, target in zip(batch_df["slice_file_name"], batch_df["classID"]):
#                 audio_path = os.path.join(pre_path, f"fold{fold}", file_path)
#                 audio, _ = librosa.load(audio_path, sr=32000, mono=True)
#                 audio = torch.tensor(audio)
#                 if audio.shape[-1] < audio_length:
#                     repeat_times = math.ceil(audio_length / audio.shape[-1])
#                     audio = audio.repeat(repeat_times)
#                     audio = audio[:audio_length]
#                 elif audio.shape[-1] > audio_length:
#                     audio = audio[:audio_length]
#                 batch_audio.append(audio)
#                 batch_targets.append(target)
            
#             batch_audio = torch.stack(batch_audio).to(device)
#             batch_targets = torch.tensor(batch_targets)
#             one_hot_targets = F.one_hot(batch_targets, num_classes=len(classes)).float()
            
#             # Process batch
#             audio_emb = model.encode_audio(batch_audio)
#             similarity = audio_emb @ text_embeds.t()
#             y_pred = F.softmax(similarity, dim=1)
            
#             y_preds.append(y_pred.cpu())
#             y_labels.append(one_hot_targets.cpu())
        
#         y_labels = torch.cat(y_labels, dim=0).numpy()
#         y_preds = torch.cat(y_preds, dim=0).numpy()
        
#         acc = accuracy_score(np.argmax(y_labels, axis=1), np.argmax(y_preds, axis=1))
#         print(f'Fold {fold} Accuracy {acc}')
#         fold_acc.append(acc)

# print(f'Urbansound8K Accuracy {np.mean(fold_acc)}')