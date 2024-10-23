#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# AudioSetCaps: An Enriched Audio-Caption Dataset using Automated Generation Pipeline with Large Audio and Language Models
# Jisheng Bai, Haohe Liu
# Northwestern Polytechnical University, Xi'an Lianfeng Acoustic Technologies Co., Ltd.
# CVSSP, University of Surrey

import argparse
import torch
from pprint import PrettyPrinter
import platform
import ruamel.yaml as YAML
from loguru import logger
from data_handling.datamodule import AudioCaptionDataModule
from models.bart_captioning import BartCaptionModel
from models.bert_captioning import BertCaptionModel
from tools.utils import setup_seed
from tqdm import tqdm
from eval_metrics import evaluate_metrics

def decode_output(predicted_output, ref_captions, file_names, epoch, beam_size=1):

    if beam_size != 1:
        caption_logger = logger.bind(indent=3)
        caption_logger.info('Captions start')
        caption_logger.info('Beam search:')
    else:
        caption_logger = logger.bind(indent=2)
        caption_logger.info('Captions start')
        caption_logger.info('Greedy search:')

    captions_pred, captions_gt, f_names = [], [], []

    for pred_cap, gt_caps, f_name in zip(predicted_output, ref_captions, file_names):

        f_names.append(f_name)
        captions_pred.append({'file_name': f_name, 'caption_predicted': pred_cap})
        ref_caps_dict = {'file_name': f_name}
        for i, cap in enumerate(gt_caps):
            ref_caps_dict[f"caption_{i + 1}"] = cap
        captions_gt.append(ref_caps_dict)

    return captions_pred, captions_gt

def validate(data_loader, model, device, epoch, beam_size):

    model.eval()
    with torch.no_grad():
        y_hat_all = []
        ref_captions_dict = []
        file_names_all = []

        for i, batch_data in tqdm(enumerate(data_loader), total=len(data_loader)):
            audios, caption_dict, audio_names, audio_ids = batch_data
            # move data to GPU
            audios = audios.to(device)

            output = model.generate(samples=audios,
                                    num_beams=beam_size)

            y_hat_all.extend(output)
            ref_captions_dict.extend(caption_dict)
            file_names_all.extend(audio_names)

        captions_pred, captions_gt = decode_output(y_hat_all, ref_captions_dict, file_names_all, epoch, beam_size=beam_size)
        metrics = evaluate_metrics(captions_pred, captions_gt)

        return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./eval_settings.yaml", type=str,
                        help="Setting files")
    
    args = parser.parse_args()
    
    yaml = YAML.YAML(typ='rt')
    with open(args.config, "r") as f:
        config = yaml.load(f)
    
    # setup distribution mode
    device = torch.device(config["device"])
    
    # setup seed
    seed = config["seed"]
    setup_seed(seed)

    main_logger = logger.bind(indent=1)

    # set up model
    device, device_name = ('cuda',
                           torch.cuda.get_device_name(torch.cuda.current_device())) \
        if torch.cuda.is_available() else ('cpu', platform.processor())
    main_logger.info(f'Process on {device_name}')

    # data loading
    datamodule = AudioCaptionDataModule(config, config["data_args"]["dataset"])
    test_loader = datamodule.test_dataloader()

    if "bart" in config["text_decoder_args"]["name"]:
        model = BartCaptionModel(config)
    elif "bert" in config["text_decoder_args"]["name"]:
        model = BertCaptionModel(config)
        
    ckpt = torch.load(config["ckpt_path"])
    print("best epoch {}".format(ckpt['epoch']))
    model.load_state_dict(ckpt["model"])
    main_logger.info(f"Loaded weights from {config['ckpt_path']}")
    device = torch.device(config["device"])
    model.to(device)
    best_beam_size = ckpt['beam_size']

    # print training settings
    printer = PrettyPrinter()
    main_logger.info('Eval:\n'
                     f'{printer.pformat(config)}')


    main_logger.info(f'Total numer of parameters: {sum([i.numel() for i in model.parameters()])}')

    main_logger.info(f'Size of test set: {len(test_loader.dataset)}, size of batches: {len(test_loader)}')

    # training loop
    main_logger.info('###### Test start ######') 
    # test after each epoch
    main_logger.info("###### Test on {} ######".format(config["data_args"]["dataset"]))
    metrics = validate(test_loader,
                       model,
                       device=device,
                       epoch=0,
                       beam_size=best_beam_size )

    best_scores = {'bleu_1': metrics["bleu_1"]["score"], 'bleu_4': metrics["bleu_4"]["score"], 'rouge_l': metrics["rouge_l"]["score"], 
                   'meteor': metrics["meteor"]["score"], 'cider': metrics['cider']['score'],
                   'spice': metrics["spice"]["score"], "spider": metrics["spider"]["score"]}
    main_logger.info('###### Best Scores {} ######'.format(best_scores))


if __name__ == '__main__':
    main()
