#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# AudioSetCaps: An Enriched Audio-Caption Dataset using Automated Generation Pipeline with Large Audio and Language Models
# Jisheng Bai, Haohe Liu
# Northwestern Polytechnical University, Xi'an Lianfeng Acoustic Technologies Co., Ltd.
# CVSSP, University of Surrey

from pprint import PrettyPrinter
import torch
import argparse
import yaml
from tqdm import tqdm
from loguru import logger
from data_handling.datamodule import AudioCaptionDataModule
from models.ase_model import ASE
from tools.utils import t2a, a2t, setup_seed
 

@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    audio_embeds_all, text_embeds_all = [], []
    for batch_idx, (audio, text, idx) in tqdm(enumerate(dataloader), total=len(dataloader)):
        audio = audio.to(device)

        audio_embeds = model.encode_audio(audio)
        text_embeds = model.encode_text(text)

        audio_embeds_all.append(audio_embeds.cpu())
        text_embeds_all.append(text_embeds.cpu())

    audio_embeds_all = torch.cat(audio_embeds_all, dim=0).numpy()
    text_embeds_all = torch.cat(text_embeds_all, dim=0).numpy()

    # evaluate text to audio retrieval
    r1, r5, r10, r50, medr, meanr, mAP10 = t2a(audio_embeds_all, text_embeds_all)

    # evaluate audio to text retrieval
    r1_a, r5_a, r10_a, r50_a, medr_a, meanr_a, mAP10_a = a2t(audio_embeds_all, text_embeds_all)

    return {"t2a": [r1, r5, r10, r50, medr, meanr, mAP10],
            "a2t": [r1_a, r5_a, r10_a, r50_a, medr_a, meanr_a, mAP10_a]}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./inference.yaml", type=str,
                        help="Setting files")

    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)

    # setup distribution mode
    device = torch.device(config["device"])

    # setup seed
    seed = config["seed"]
    setup_seed(seed)

    dataset_name = config["data_args"]["dataset"]
    
    # load evaluation datamodule
    datamodule = AudioCaptionDataModule(config, dataset_name)
    test_loader = datamodule.test_dataloader()
    
    # setup model
    model = ASE(config)
    model = model.to(device)

    main_logger = logger.bind(indent=1)

    # print training settings
    printer = PrettyPrinter()
    main_logger.info('Eval:\n'
                     f'{printer.pformat(config)}')

    model = ASE(config)
    model.to(device)
    
    ckpt_path = config['ckpt_path']
    cp = torch.load(ckpt_path)
    model.load_state_dict(cp['model'], strict=False)
    # model.load_state_dict(cp['model'])
    model.eval()
    main_logger.info(f"Loaded weights from {config['ckpt_path']}")
    main_logger.info(f'Total numer of parameters: {sum([i.numel() for i in model.parameters()])}')

    # eval
    metrics = validate(model, test_loader, device)

    main_logger.info('###### Eval on {} done ######'.format(dataset_name))
    main_logger.info('###### Best Metrics {} ######'.format(metrics))
    


if __name__ == '__main__':
    main()
