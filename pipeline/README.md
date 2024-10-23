## Codes for audio caption data generation pipeline

## Dataset
Prepare the dataset to be captioned.
Must prepare a csv file including the filenames and labels.

## Run the code
Step 1: create environment for qwen-audio  
```
conda create -n qwen python=3.10
pip install -r qwen_requirement.txt
conda install FFmpeg
```  
Step 2: install [flash attention](https://github.com/Dao-AILab/flash-attention) to boost the procedure  
Step 3: create environment for CLAP  
```
conda create -n clap python=3.10
pip install laion-clap
git clone https://github.com/LAION-AI/CLAP.git
cd CLAP
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```  
Step 4: install [ollama](https://github.com/Dao-AILab/flash-attention) `curl https://ollama.ai/install.sh | sh`  
Step 5: change the conda.sh path in Audiosetcaps_qwen_main.sh and Audiosetcaps_Mistral_main.sh  
Step 6: run `bash Audiosetcaps_qwen_main.sh` to generate qwen-audio captions  
Step 7: `ollama serve`  
Step 8: `ollama run mistral`  
Step 9: download CLAP ckpt [music_speech_audioset_epoch_15_esc_89.98.pt](https://huggingface.co/lukewys/laion_clap/blob/main/music_speech_audioset_epoch_15_esc_89.98.pt)  
Step 10: set paths and parameters in `config.py`  
Step 11: run `bash Audiosetcaps_Mistral_main.sh` to generate Mistral captions





