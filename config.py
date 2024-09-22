# -*- coding: utf-8 -*-

### general
gpu_id = '0'
root_audio_path = r'~/audio' # root path for audio clips
as_csv_path = r'~/filename.csv' # csv path including filename and label

### Qwen-audio
qwen_caption_path = r'~/qwen_caption_save_path' # path for saving qwen-auido captions
model_id = 'qwen/Qwen-Audio-Chat'
revision = 'master'
cache_dir = r'~/Audioset_caption/' # path for downloading qwen-auido ckpts
qwen_try_num = 2

### Mistral
mistral_try_num = 2             # repeat number of refinement
default_prompt_path = r'~/default_prompt.txt' # mistral prompt
mis_caption_path = r'~/Mistral_caption_save_path' # path for saving Mistral captions

### CLAP
pretrained_path = r'~/music_speech_audioset_epoch_15_esc_89.98.pt' # pre-trained CLAP ckpt path
CLAP_score_csv_path = r'~/caption_scores.csv' # csv path for saving captions scores from CLAP
