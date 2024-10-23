## Evaluate AudioSetCaps Fine-tuned Models on Automated Audio Captioning
* Environmental settings please refers to [WavCaps](https://github.com/XinhaoMei/WavCaps)

### Pre-trained Models on AudioSetCaps
* Pre-trained model using AudioSetCaps can be downloaded at [Hugging Face](https://huggingface.co/datasets/baijs/AudioSetCaps/blob/main/Model/AAC_HTSAT%2BBART_ASC_PT.pt)
* Pre-trained model using AudioSetCaps+AudioCaps+Clotho can be downloaded at [Hugging Face](https://huggingface.co/datasets/baijs/AudioSetCaps/blob/main/Model/AAC_HTSAT%2BBART_ASC%2BAC%2BClotho_PT.pt)

### Fine-tuned Models on AudioSetCaps
* Fine-tuned model with AudioSetCaps pre-training can be downloaded at [Hugging Face](https://huggingface.co/datasets/baijs/AudioSetCaps/blob/main/Model/AAC_HTSAT%2BBART_ASC_PT_AC_FT_CR83.9_SR51.3.pt)
* Fine-tuned model with AudioSetCaps+AudioCaps+Clotho pre-training can be downloaded at [Hugging Face](Model/AAC_HTSAT+BART_ASC+AC+Clotho_PT_AC_FT_CR84.8_SR51.6.pt)

### Download pretrained audio encoders
* Please download pretrained audio encoder [HTSAT](https://drive.google.com/drive/folders/1ZaYERuMMLLgu4oHTl47FcippLFboaGq5?usp=share_link).

* Put them under `pretrained_models/audio_encoders`.

### Configuration
* Download fine-tuned AAC model and change evaluation settings in yaml files `eval_settings.yaml`.

* Please prepare COCO caption evaluation tools in `coco_caption` folder.

* Please prepare AudioCaps `train.json` `val.json` `test.json` in `data/AudioCaps/json_files` folder.

* Run `eval.py` for evaluation.