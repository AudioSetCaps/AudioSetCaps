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
from data_handling.text_transform import text_preprocess
from models.ase_model import ASE
import torch.nn.functional as F
import os
import math



##### load model
config = yaml.load(open("./inference.yaml", "r"), Loader=yaml.FullLoader)
device = "cuda"

model = ASE(config)
model.to(device)
ckpt_path = config['ckpt_path']
cp = torch.load(ckpt_path)
model.load_state_dict(cp['model'], strict=False)
model.eval()
print("Model weights loaded from {}".format(ckpt_path))


######## CommonLanguage
print("CommonLanguage ZSC")
CL_df = pd.read_csv(r'data/CommonLanguage/CommonLanguage_test.csv')
labels = sorted(CL_df['Language'].unique())
classes = ["The language showcases {}".format(text_preprocess(x)) for x in labels]

print(classes[:10])

batch_size = 32
audio_length = 32000 * 10

with torch.no_grad():
    text_embeds = model.encode_text(classes)
    y_preds, y_labels = [], []

    for i in tqdm(range(0, len(CL_df), batch_size), desc="Processing batches"):
        batch_df = CL_df.iloc[i:i+batch_size]
        batch_audio = []
        batch_targets = []

        for _, row in batch_df.iterrows():
            file_path = row['File Path']
            language = row['Language']
            if os.path.exists(file_path):
                audio, _ = librosa.load(file_path, sr=32000, mono=True)
                audio = torch.tensor(audio)
                if audio.shape[-1] < audio_length:
                    repeat_times = math.ceil(audio_length / audio.shape[-1])
                    audio = audio.repeat(repeat_times)
                    audio = audio[:audio_length]
                elif audio.shape[-1] > audio_length:
                    audio = audio[:audio_length]
                batch_audio.append(audio)
                batch_targets.append(labels.index(language))
            else:
                continue

        if not batch_audio:
            continue

        batch_audio = torch.stack(batch_audio).to(device)
        batch_targets = torch.tensor(batch_targets)
        one_hot_targets = F.one_hot(batch_targets, num_classes=len(classes)).float()

        audio_emb = model.encode_audio(batch_audio)
        similarity = audio_emb @ text_embeds.t()
        y_pred = F.softmax(similarity, dim=1)

        y_preds.append(y_pred.cpu())
        y_labels.append(one_hot_targets.cpu())

    y_labels = torch.cat(y_labels, dim=0).numpy()
    y_preds = torch.cat(y_preds, dim=0).numpy()
    acc = accuracy_score(np.argmax(y_labels, axis=1), np.argmax(y_preds, axis=1))
print(f'CommonLanguage Accuracy {acc}')


######## CREMA-D
# print("CREMA-D ZSC")
# CREMA_D_df = pd.read_csv(r'data/CREMA-D/CREMA-D_test.csv')
# labels = sorted(CREMA_D_df['Emotion'].unique())
# classes = ["This clip is {} feeling".format(text_preprocess(x)) for x in labels]

# print(classes[:10])

# batch_size = 32
# audio_length = 32000 * 10

# with torch.no_grad():
#     text_embeds = model.encode_text(classes)
#     y_preds, y_labels = [], []

#     for i in tqdm(range(0, len(CREMA_D_df), batch_size), desc="Processing batches"):
#         batch_df = CREMA_D_df.iloc[i:i+batch_size]
#         batch_audio = []
#         batch_targets = []

#         for _, row in batch_df.iterrows():
#             file_path = row['Audio Path']
#             emotion = row['Emotion']
#             if os.path.exists(file_path):
#                 audio, _ = librosa.load(file_path, sr=32000, mono=True)
#                 audio = torch.tensor(audio)
#                 if audio.shape[-1] < audio_length:
#                     repeat_times = math.ceil(audio_length / audio.shape[-1])
#                     audio = audio.repeat(repeat_times)
#                     audio = audio[:audio_length]
#                 elif audio.shape[-1] > audio_length:
#                     audio = audio[:audio_length]
#                 batch_audio.append(audio)
#                 batch_targets.append(labels.index(emotion))
#             else:
#                 continue

#         if not batch_audio:
#             continue

#         batch_audio = torch.stack(batch_audio).to(device)
#         batch_targets = torch.tensor(batch_targets)
#         one_hot_targets = F.one_hot(batch_targets, num_classes=len(classes)).float()

#         audio_emb = model.encode_audio(batch_audio)
#         similarity = audio_emb @ text_embeds.t()
#         y_pred = F.softmax(similarity, dim=1)

#         y_preds.append(y_pred.cpu())
#         y_labels.append(one_hot_targets.cpu())

#     y_labels = torch.cat(y_labels, dim=0).numpy()
#     y_preds = torch.cat(y_preds, dim=0).numpy()
#     acc = accuracy_score(np.argmax(y_labels, axis=1), np.argmax(y_preds, axis=1))
# print(f'CREMA-D Accuracy {acc}')


# ######## RAVDESS
# print("RAVDESS ZSC")
# RAVDESS_df = pd.read_csv(r'data/RAVDESS/RAVDESS_test.csv')
# labels = sorted(RAVDESS_df['Emotion'].unique())
# classes = ["This emotion is " + text_preprocess(l) for l in labels]

# print(classes[:10])

# batch_size = 32
# audio_length = 32000 * 10

# with torch.no_grad():
#     text_embeds = model.encode_text(classes)
#     y_preds, y_labels = [], []

#     for i in tqdm(range(0, len(RAVDESS_df), batch_size), desc="Processing batches"):
#         batch_df = RAVDESS_df.iloc[i:i+batch_size]
#         batch_audio = []
#         batch_targets = []

#         for _, row in batch_df.iterrows():
#             file_path = row['Audio Path']
#             emotion = row['Emotion']
#             if os.path.exists(file_path):
#                 audio, _ = librosa.load(file_path, sr=32000, mono=True)
#                 audio = torch.tensor(audio)
#                 if audio.shape[-1] < audio_length:
#                     repeat_times = math.ceil(audio_length / audio.shape[-1])
#                     audio = audio.repeat(repeat_times)
#                     audio = audio[:audio_length]
#                 elif audio.shape[-1] > audio_length:
#                     audio = audio[:audio_length]
#                 batch_audio.append(audio)
#                 batch_targets.append(labels.index(emotion))
#             else:
#                 continue

#         if not batch_audio:
#             continue

#         batch_audio = torch.stack(batch_audio).to(device)
#         batch_targets = torch.tensor(batch_targets)
#         one_hot_targets = F.one_hot(batch_targets, num_classes=len(classes)).float()

#         audio_emb = model.encode_audio(batch_audio)
#         similarity = audio_emb @ text_embeds.t()
#         y_pred = F.softmax(similarity, dim=1)

#         y_preds.append(y_pred.cpu())
#         y_labels.append(one_hot_targets.cpu())

#     y_labels = torch.cat(y_labels, dim=0).numpy()
#     y_preds = torch.cat(y_preds, dim=0).numpy()
#     acc = accuracy_score(np.argmax(y_labels, axis=1), np.argmax(y_preds, axis=1))
# print(f'RAVDESS Accuracy {acc}')


# ######## GTZAN
# print("GTZAN ZSC")
# GTZAN_df = pd.read_csv(r'data/GTZAN/GTZAN_test.csv')
# labels = sorted(GTZAN_df['Music Genre'].unique())
# classes = ["{} genre can be heard".format(text_preprocess(x)) for x in labels]

# print(classes[:10])

# batch_size = 32
# audio_length = 32000 * 10

# with torch.no_grad():
#     text_embeds = model.encode_text(classes)
#     y_preds, y_labels = [], []

#     for i in tqdm(range(0, len(GTZAN_df), batch_size), desc="Processing batches"):
#         batch_df = GTZAN_df.iloc[i:i+batch_size]
#         batch_audio = []
#         batch_targets = []

#         for _, row in batch_df.iterrows():
#             file_path = row['Audio Path']
#             genre = row['Music Genre']
#             if os.path.exists(file_path):
#                 try:
#                     audio, _ = librosa.load(file_path, sr=32000, mono=True)
#                     audio = torch.tensor(audio)
#                     if audio.shape[-1] < audio_length:
#                         repeat_times = math.ceil(audio_length / audio.shape[-1])
#                         audio = audio.repeat(repeat_times)
#                         audio = audio[:audio_length]
#                     elif audio.shape[-1] > audio_length:
#                         audio = audio[:audio_length]
#                     batch_audio.append(audio)
#                     batch_targets.append(labels.index(genre))
#                 except:
#                     continue
#             else:
#                 continue

#         if not batch_audio:
#             continue

#         batch_audio = torch.stack(batch_audio).to(device)
#         batch_targets = torch.tensor(batch_targets)
#         one_hot_targets = F.one_hot(batch_targets, num_classes=len(classes)).float()

#         audio_emb = model.encode_audio(batch_audio)
#         similarity = audio_emb @ text_embeds.t()
#         y_pred = F.softmax(similarity, dim=1)

#         y_preds.append(y_pred.cpu())
#         y_labels.append(one_hot_targets.cpu())

#     total_predictions = sum(len(pred) for pred in y_preds)
#     print(f"predict {total_predictions} recordings")

#     y_labels = torch.cat(y_labels, dim=0).numpy()
#     y_preds = torch.cat(y_preds, dim=0).numpy()
#     acc = accuracy_score(np.argmax(y_labels, axis=1), np.argmax(y_preds, axis=1))
# print(f'GTZAN Accuracy {acc}')
    

# ######## openmic-2018
# print("openmic-2018 ZSC")
# openmic_df = pd.read_csv(r'data/OpenMIC-2018/OpenMIC-2018_test.csv')
# labels = sorted(openmic_df['Instrument'].unique())
# classes = ["This music clip showcases {} instrument".format(x.replace('_', ' ')) for x in labels]


# print(classes[:10])

# batch_size = 32
# audio_length = 32000 * 10

# with torch.no_grad():
#     text_embeds = model.encode_text(classes)
#     y_preds, y_labels = [], []

#     for i in tqdm(range(0, len(openmic_df), batch_size), desc="Processing batches"):
#         batch_df = openmic_df.iloc[i:i+batch_size]
#         batch_audio = []
#         batch_targets = []

#         for _, row in batch_df.iterrows():
#             file_path = row['Audio Path']
#             instrument = row['Instrument']
#             if os.path.exists(file_path):
#                 try:
#                     audio, _ = librosa.load(file_path, sr=32000, mono=True)
#                     audio = torch.tensor(audio)
#                     if audio.shape[-1] < audio_length:
#                         repeat_times = math.ceil(audio_length / audio.shape[-1])
#                         audio = audio.repeat(repeat_times)
#                         audio = audio[:audio_length]
#                     elif audio.shape[-1] > audio_length:
#                         audio = audio[:audio_length]
#                     batch_audio.append(audio)
#                     batch_targets.append(labels.index(instrument))
#                 except:
#                     continue
#             else:
#                 continue

#         if not batch_audio:
#             continue

#         batch_audio = torch.stack(batch_audio).to(device)
#         batch_targets = torch.tensor(batch_targets)
#         one_hot_targets = F.one_hot(batch_targets, num_classes=len(classes)).float()

#         audio_emb = model.encode_audio(batch_audio)
#         similarity = audio_emb @ text_embeds.t()
#         y_pred = F.softmax(similarity, dim=1)

#         y_preds.append(y_pred.cpu())
#         y_labels.append(one_hot_targets.cpu())

#     print(f"predict {sum(len(pred) for pred in y_preds)} recordings")
    
#     y_labels = torch.cat(y_labels, dim=0).numpy()
#     y_preds = torch.cat(y_preds, dim=0).numpy()
#     acc = accuracy_score(np.argmax(y_labels, axis=1), np.argmax(y_preds, axis=1))

# print(f'openmic 2018 Accuracy {acc}')
