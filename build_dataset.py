import sys
from typing import List

import numpy as np
import torch
import tritonclient.grpc as grpcclient
import torchaudio
from datasets import load_dataset
import json


class CosyVoice:
    def __init__(self, tritonserver_base):
        self.triton_client = grpcclient.InferenceServerClient(url=tritonserver_base)

    def preprocess_prompt_audio(self, audio_path: str):
        audio, sr = torchaudio.load(audio_path)
        audio = audio[:, :(audio.shape[1] // sr) * sr]
        print(audio.shape)
        if audio.shape[1] // sr < 1:
            return None
        audio_16k = torchaudio.functional.resample(audio, sr, 16000).cpu().numpy()
        audio_24k = torchaudio.functional.resample(audio, sr, 24000)
        client = self.triton_client
        # Create inputs for the model
        audio_input = grpcclient.InferInput("audio", audio_16k.shape, "FP32")
        audio_input.set_data_from_numpy(audio_16k)

        # Prepare the output request
        speech_feat_output = grpcclient.InferRequestedOutput("speech_feat")
        embedding_output = grpcclient.InferRequestedOutput("embedding")
        prompt_speech_token_output = grpcclient.InferRequestedOutput("prompt_speech_token")

        # Make the inference request
        response = client.infer(
            model_name='ensembled_feature_extractor',
            inputs=[audio_input],
            outputs=[speech_feat_output, embedding_output, prompt_speech_token_output]
        )

        # Extract the output data
        embedding = response.as_numpy("embedding")
        speech_token = response.as_numpy("prompt_speech_token")
        torchaudio.save(audio_path + '.24k.wav', audio_24k, 24000, encoding="PCM_S", bits_per_sample=16)
        return audio_path + '.24k.wav', speech_token.tolist(), embedding.tolist()


    def prepare_ds(self, file_path):
        data = load_dataset('json', data_files=file_path, split='train')
        with open(file_path + '.vocos.json', 'w', encoding='utf8') as f:
            for x in data['audio']:
                r = self.preprocess_prompt_audio(x)
                if r is not None:
                    tmp = {"audio": r[0], 'speech_token': r[1], 'embedding': r[1]}
                    f.write(json.dumps(tmp, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    m = CosyVoice('0.0.0.0:8001')
    m.prepare_ds(sys.argv[1])
