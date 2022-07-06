# Copyright (c) 2022 NVIDIA CORPORATION. 
#   Licensed under the MIT license.

import os
import sys
from collections import defaultdict
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy.io import wavfile 

from pesq import pesq
from pystoi import stoi


def evaluate_dns(testset_path, enhanced_path, target):
    reverb = 'no'
    result = defaultdict(int)

    for i in tqdm(range(300)):
        try:
            rate, clean = wavfile.read(os.path.join(testset_path, "clean", "clean_fileid_{}.wav".format(i)))
            if target == 'noisy':
                rate, target_wav = wavfile.read(os.path.join(testset_path, "noisy", "noisy_fileid_{}.wav".format(i)))
            else:
                rate, target_wav = wavfile.read(os.path.join(enhanced_path, "enhanced_fileid_{}.wav".format(i)))
        except:
            continue

        length = target_wav.shape[-1]

        result['pesq_wb'] += pesq(16000, clean, target_wav, 'wb') * length  # wide band
        result['pesq_nb'] += pesq(16000, clean, target_wav, 'nb') * length  # narrow band
        result['stoi'] += stoi(clean, target_wav, rate) * length
        result['count'] += 1 * length
    
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='dns', help='dataset')
    parser.add_argument('-e', '--enhanced_path', type=str, help='enhanced audio path')
    parser.add_argument('-t', '--testset_path', type=str, help='testset path')
    args = parser.parse_args()

    enhanced_path = args.enhanced_path
    testset_path = args.testset_path
    target = 'enhanced'

    if args.dataset == 'dns':
        result = evaluate_dns(testset_path, enhanced_path, target)
        
    # logging
    for key in result:
        if key != 'count':
            print('{} = {:.3f}'.format(key, result[key]/result['count']), end=", ")
