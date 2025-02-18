from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
from snac import SNAC
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
    tokenizer = AutoTokenizer.from_pretrained(sys.argv[1])
    model = AutoModelForCausalLM.from_pretrained(sys.argv[2], torch_dtype=torch.bfloat16).to('cuda')
    snac_model = SnacVocos.from_pretrained('/home/zhou/data3/tts/vocos/logs/lightning_logs/version_3').eval().to('cuda')

    def decode(l):
        first_elements, second_elements, third_elements = split_sequence(l)
        codes = [torch.from_numpy(np.array(x).astype(np.int32)[None,]).to('cuda') for x in
                 [first_elements, second_elements, third_elements]]
        audio_hat = snac_model.decode(codes)
        z_q = snac_model.quantizer.from_codes(codes)
        return audio_hat, z_q

    def prepare_input(text, prompt):
        text_ids = tokenizer(text, add_special_tokens=False).input_ids
        input_ids = prompt + [155999, 156010, 156005] + text_ids + [156006, 156007]
        input_ids = torch.tensor([input_ids], dtype=torch.int64).to('cuda')
        return input_ids

    with torch.no_grad():
        passages = ["What can I do for you, sir?","是吧?我也觉得他们这个project不会成功的。","这样的做法我认为是可行的,不用太担心。", "我们的C-E-O是杜晓祥。"]
        passages = [x.lower() for x in passages]
        prompts = [
                [156009,155747,155777,155830,155868,155872,155918,155950,155987,156010,156011],
[156009,155753,155791,155833,155842,155878,155921,155963,155978,156010,156011],
[156009,155753,155802,155825,155848,155892,155902,155963,155980,156010,156011],
[156009,155750,155795,155833,155864,155881,155908,155963,155969,156010,156011],
[156009,155767,155789,155809,155866,155893,155915,155956,155992,156010,156011],
[156009,155757,155776,155806,155846,155892,155919,155942,155980,156010,156011],
[156009,155767,155788,155820,155859,155875,155916,155939,155977,156010,156011],
            [156009, 155757, 155783, 155827, 155856, 155874, 155909, 155939, 155986, 156010, 156011],
            [156009, 155760, 155793, 155823, 155856, 155874, 155925, 155945, 155968, 156010, 156011]]
        for i, prompt in enumerate(prompts[::-1]):
            try:
                z = []
                for x in passages:
                    if len(x) == 0:
                        break
                    x = prepare_input(x.strip(), prompt)
                    output_ids = model.generate(x, eos_token_id=156008, no_repeat_ngram_size=0, num_beams=1,
                                                do_sample=False, repetition_penalty=1.2,
                                                suppress_tokens=list(range(151641)))
                    output_ids = output_ids[0, x.shape[-1]:].cpu().numpy().tolist()
                    output_ids = [x for x in output_ids if x not in {156013, 156008}]
                    z = z + output_ids
                output_ids = [int(tokenizer.convert_ids_to_tokens(x).replace('<|speech-', '').replace('|>', '')) for x
                              in
                              z]
                with torch.no_grad():
                    audio_hat_all = snac_model.split_sequence(output_ids)
                    torchaudio.save('test_{}.wav'.format(i), audio_hat_all.cpu(), 24000)
            except Exception as e:
                print(e)


if __name__ == '__main__':
    test()
