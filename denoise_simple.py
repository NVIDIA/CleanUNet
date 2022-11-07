# Adapted from https://github.com/NVIDIA/waveglow under the BSD 3-Clause License.

# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

import os
import argparse
import json
from tqdm import tqdm
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter

import random
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

from scipy.io.wavfile import write as wavwrite

from dataset import load_CleanNoisyPairDataset
from util import rescale, find_max_epoch, print_size, sampling
from network import CleanUNet
import torchaudio

def load_simple(filename):
    audio, _ = torchaudio.load(filename)
    return audio
    


def denoise(files, ckpt_path):
    """
    Denoise audio

    Parameters:
    output_directory (str):         save generated speeches to this path
    ckpt_iter (int or 'max'):       the pretrained checkpoint to be loaded; 
                                    automitically selects the maximum iteration if 'max' is selected
    subset (str):                   training, testing, validation
    dump (bool):                    whether save enhanced (denoised) audio
    """

    # setup local experiment path
    exp_path = train_config["exp_path"]
    print('exp_path:', exp_path)

    # load data
    loader_config = deepcopy(trainset_config)
    loader_config["crop_length_sec"] = 0



    # predefine model
    net = CleanUNet(**network_config).cuda()
    print_size(net)

    # load checkpoint
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()

    # inference
    batch_size = 1000000
    for file_path in tqdm(files):
        file_name = os.path.basename(file_path)
        file_dir = os.path.dirname(file_name)
        new_file_name = file_name + "_denoised.wav"
        noisy_audio = load_simple(file_path).cuda()
        LENGTH = len(noisy_audio[0].squeeze())
        noisy_audio = torch.chunk(noisy_audio, LENGTH // batch_size + 1, dim=1)
        all_audio = []

        for batch in tqdm(noisy_audio):
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    generated_audio = sampling(net, batch)
                    generated_audio = generated_audio.cpu().numpy().squeeze()
                    all_audio.append(generated_audio)
        
        all_audio = np.concatenate(all_audio, axis=0)
        save_file = os.path.join(file_dir, new_file_name)
        print("saved to:", save_file)
        wavwrite(save_file, 
                32000,
                all_audio.squeeze())



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config.json',
                        help='JSON file for configuration')
    parser.add_argument('-ckpt_iter', '--ckpt_iter', default='max',
                        help='Which checkpoint to use; assign a number or "max" or "pretrained"')     

    parser.add_argument('files', nargs=argparse.REMAINDER)

    args = parser.parse_args()

    # Parse configs. Globals nicer in this case
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    gen_config              = config["gen_config"]
    global network_config
    network_config          = config["network_config"]      # to define wavenet
    global train_config
    train_config            = config["train_config"]        # train config
    global trainset_config
    trainset_config         = config["trainset_config"]     # to read trainset configurations
    files = args.files

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    denoise(files,args.ckpt_iter)
    
