## Evaluate AudioSetCaps Fine-tuned Models on Audio-Text Retrieval and Zero-shot Classification
* Environmental settings please refers to [WavCaps](https://github.com/XinhaoMei/WavCaps)

### Pre-trained Model on AudioSetCaps
* Pre-trained model using AudioSetCaps with 25% text mask can be downloaded at [Hugging Face](https://huggingface.co/datasets/baijs/AudioSetCaps/blob/main/Model/ATR_HTSAT%2BRoBERTa_ASC_PT_mask25_T2AR1_32.8_A2TR1_44.5.pt)

### Fine-tuned Model on AudioSetCaps
* Fine-tuned model on AudioCaps using AudioSetCaps with 25% text mask pre-training can be downloaded at [Hugging Face](https://huggingface.co/datasets/baijs/AudioSetCaps/blob/main/Model/ATR_HTSAT%2BRoBERTa_ASC_PT_mask25_AC_FT_T2AR1_46.3_A2TR1_59.3.pt)

### Pre-trained Model for Zero-shot Classification
* Pre-trained using AudioSetCaps+AudioCaps+Clotho with 25% text mask can be downloaded at [Hugging Face](https://huggingface.co/datasets/baijs/AudioSetCaps/blob/main/Model/ZSC_HTSAT%2BRoBERTa_ASC%2BAC%2BClotho_PT_mask25_T2AR1_41.0_A2TR1_55.6.pt)

### Download pretrained audio encoders
* Please download pretrained audio encoder [HTSAT](https://drive.google.com/drive/folders/1ZaYERuMMLLgu4oHTl47FcippLFboaGq5?usp=share_link).

* Put them under `pretrained_models/audio_encoders`.

### Configuration
* Download fine-tuned ATR models and change evaluation settings in yaml files `inference.yaml`.

* Please prepare AudioCaps `train.json` `val.json` `test.json` in `data/AudioCaps/json_files` folder.

* Please prepare zero-shot classification datasets and csvs (file names and labels for some tasks are given) in `data/` folder.

* Run `eval.py` for evaluation on AudioCaps.

* Run `zero_shot_classification_environmental_sound.py` for environmental sound zero-shot classification tasks.

* Run `zero_shot_classification_speech_music.py` for speech and music zero-shot classification tasks.