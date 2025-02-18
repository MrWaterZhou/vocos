from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import torch
import numpy as np
import torchaudio
from vocos import SnacVocos


def split_sequence(sequence):
    group_size = 7
    first_elements = []
    second_elements = []
    third_elements = []

    # Iterate over the sequence in chunks of 7
    for i in range(0, len(sequence), group_size):
        group = sequence[i:i + group_size]

        # Add elements to the respective lists based on their position in the group
        if len(group) >= 1:
            first_elements.append(group[0])
        if len(group) >= 5:
            second_elements.extend([group[1], group[4]])
        if len(group) >= 7:
            third_elements.extend([group[2], group[3], group[5], group[6]])
        else:
            third_elements.extend(group[2:])

    return first_elements, second_elements, third_elements


def sliding_window(data, window_size=7, step=7):
    return [data[i:i + window_size] for i in range(0, len(data) - window_size + 1, step)]


def test():
    from datasets import load_dataset
    snac_model = SnacVocos.from_pretrained('/home/zhou/data3/tts/vocos/logs/lightning_logs/version_1').eval().to('cuda')
    ds = load_dataset('json', data_files=sys.argv[1], split='train').take(100)
    for i, target_id in enumerate(ds['target']):
        audio_hat_all = snac_model.split_sequence(target_id)
        torchaudio.save('test_{}.wav'.format(i), audio_hat_all.cpu(), 24000)


if __name__ == '__main__':
    test()
