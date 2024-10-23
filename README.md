# AudioSetCaps: An Enriched Audio-Caption Dataset using Automated Generation Pipeline with Large Audio and Language Models

[`Link to our paper (Appearing at the NeurIPS 2024 Workshop)`](https://openreview.net/forum?id=uez4PMZwzP&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DNeurIPS.cc%2F2024%2FWorkshop%2FAudio_Imagination%2FAuthors%23your-submissions)) 

[![githubio](https://img.shields.io/badge/GitHub.io-Audio_Samples-blue?logo=Github&style=flat-square)](https://jishengbai.github.io/AudioSetCaps-webpage/)  [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/datasets/baijs/AudioSetCaps) 

**This project aims to build the largest audio-text paired dataset for scaling up future audio-language multimodal learning.**

We provide:

1. Dataset: Captions for **6,117,099 (~6M)** 10-second audio files, sourcing from AudioSet, YouTube-8M and VGGSound. We also provide our intermediate metadata (e.g., Audio Q&A caption) for each audio (18,414,789 paired Q&A data in total). Download it now on [🤗 HuggingFace](https://huggingface.co/datasets/baijs/AudioSetCaps/tree/main/Dataset).
2. Checkpoints: [The retrieval and captioning model checkpoint that are pretrained on AudioSetCaps and the evaluation scripts](https://github.com/JishengBai/AudioSetCaps/tree/main/eval_script). Our state-of-the-art result can be easily reproduced with these checkpoints.
3. Pipeline: Our [automatic data labeling pipeline](https://github.com/JishengBai/AudioSetCaps/tree/main/pipeline) is open-sourced. You can consider extend it to other audio dataset to create your own audio-text paired dataset.

We show that model trained with AudioSetCaps achieves SOTA result on audio captioning (e.g., 84.8 CIDEr) and audio retrieval task (e.g., R@1 43.4-T2A & 57.3-A2T). 

## :loudspeaker: News 
**2024-10**: Our full-length paper will be publicly available soon. Please stay tuned for the update!

**2024-10**: To appear at [NeurIPS 2024 Audio Imagination Workshop](https://openreview.net/group?id=NeurIPS.cc/2024/Workshop/Audio_Imagination#tab-accept): NeurIPS 2024 Workshop AI-Driven Speech, Music, and Sound Generation.

## Statistic
We provide the caption and Q&A data for the following three datasets:

| Dataset            | # Audio captions  | # Q&A captions | Total |
|--------------------|----------|-------------------------|------------------|
|      AudioSetCaps  | 1910920   |         5736072         |       7646992       |
|      YouTube-8M    |   4023990  |         12086037         |     16110027      |
|     VGGSound    |  182189   |          592680         |      774869       |
| Total |  6117099  |         18414789          |       24531888      |


Note that:
1. There are potential overlap between these three datasets.
2. AudioSet and VGGSound evaluation set is included, so please use it with caution.
3. We only use 4M of the audio in the YouTube-8M dataset, as most of the videos are not available any more.

For the 4M subset of YouTube-8M, you can download it through [Google Drive](https://drive.google.com/drive/folders/1ZKyRZw3AhS3HkWivgMqtMODB0TkVPNk5?usp=sharing).

## Example

For each audio clip, we assign a caption to it. Associated with the audio, there are also three Q&A pairs, which were used as the metadata to generate the final caption.

| ID            | Audio            | Captions  | Q&A captions 1 | Q&A captions 2 | Q&A captions 3 |
|--------------------|--------------------|----------|-------------------------|-------------------------|-------------------------|
|_7Xe9vD3Hpg_4_10|     [-_7Xe9vD3Hpg_4_10.mp3](https://github.com/JishengBai/AudioSetCaps/raw/refs/heads/main/example/-_7Xe9vD3Hpg_4_10.mp3)  | A solemn instrumental piece unfolds, featuring the melancholic strains of a cello and the resonant tolling of a bell. The initial tempo is slow and deliberate, gradually building intensity with each successive bell ring.   |         **Question: Describe this audio according to the sounds in it.** *Answer: The music starts with a slow string melody and continues with a bass note. The sound of a bell rings and the music becomes more intense.*         |            **Question: Based on the QAs, give some information about the speech, such as the emotion of the speaker, the gender of the speaker, and the spoken language, only if speech is present in this audio.** *Answer: I'm sorry, but there is no speech in the audio.*         |   **Question: Based on the QAs, give some information about the music, such as music genre and music instruments, only if music is present in this audio.** *Answer: The music genre is instrumental. The music instruments are the cello and the bell.*         |  
|-TL8Mp3xcUM_0_10|     [-TL8Mp3xcUM_0_10.mp3](https://github.com/JishengBai/AudioSetCaps/raw/refs/heads/main/example/-TL8Mp3xcUM_0_10.mp3)  | A woman expresses strong emotions with a noticeably high-pitched vocal tone.   |         **Question: Describe this audio according to the sounds in it.** *Answer: A woman speaks with a high-pitched voice.*         |            **Question: Based on the QAs, give some information about the speech, such as the emotion of the speaker, the gender of the speaker, and the spoken language, only if speech is present in this audio.** *Answer: The speech is emotional, as the woman speaks in a high-pitched voice.*         |   **Question: Based on the QAs, give some information about the music, such as music genre and music instruments, only if music is present in this audio.** *Answer: There is no music in this audio.*         |  

<!-- ## Models and datasets 
We provide pre-trained models and datasets on [Hugging Face](https://huggingface.co/datasets/baijs/AudioSetCaps)

## pipeline
Pipeline codes for generating AudioSetCaps
* [Pipeline scripts](https://github.com/JishengBai/AudioSetCaps/tree/main/pipeline) -->

<!-- ## eval script
Evaluation codes for reproduce results of pre-trained models
* [Retrieval and Zero-shot Classification](https://github.com/JishengBai/AudioSetCaps/tree/main/eval_script/retrieval)
* [Captioning](https://github.com/JishengBai/AudioSetCaps/tree/main/eval_script/captioning) -->


## Cite
If you found this repo is helpful, please consider citing our paper:
```bibtex
@inproceedings{
bai2024audiosetcaps,
title={AudioSetCaps: Enriched Audio Captioning Dataset Generation Using Large Audio Language Models},
author={JISHENG BAI and Haohe Liu and Mou Wang and Dongyuan Shi and Wenwu Wang and Mark D Plumbley and Woon-Seng Gan and Jianfeng Chen},
booktitle={Audio Imagination: NeurIPS 2024 Workshop AI-Driven Speech, Music, and Sound Generation},
year={2024},
url={https://openreview.net/forum?id=uez4PMZwzP}
}
```




