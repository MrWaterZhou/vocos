from typing import List

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torchaudio
from encodec import EncodecModel
from snac import SNAC
import whisper
from vocos.modules import safe_log


class ONNXMultiInputDynamicWrapper(nn.Module):
    def __init__(self, onnx_model_path):
        """
        使用 ONNX Runtime 的 IO Binding 处理多个输入和动态输出。
        :param onnx_model_path: ONNX 模型文件路径
        :param device: 指定设备 ('cuda' 或 'cpu')
        """
        super(ONNXMultiInputDynamicWrapper, self).__init__()
        device_id = torch.cuda.current_device()

        # 初始化 ONNX Runtime session
        providers = [('CUDAExecutionProvider', {'device_id': device_id})]
        self.session = ort.InferenceSession(onnx_model_path, providers=providers)

        # 获取输入和输出名称
        self.input_names = [inp.name for inp in self.session.get_inputs()]  # 获取两个输入
        self.output_names = [out.name for out in self.session.get_outputs()]  # 获取输出

        # 推理设备
        self.device = f'cuda:{device_id}'
        self.io_binding = self.session.io_binding()

    def forward(self, feats: torch.Tensor, feats_length: torch.Tensor):
        """
        处理两个输入张量并生成动态输出张量。
        :param feats: PyTorch Tensor, (1, 128, -1), float32
        :param feats_length: PyTorch Tensor, (1), int32
        :return: 输出 Tensor (1, -1), int32
        """
        # 确保输入 Tensor 在正确设备上
        feats = feats.to(self.device)
        feats_length = feats_length.to(self.device)

        # 绑定输入 feats
        self.io_binding.bind_input(
            name=self.input_names[0],  # 假设第一个输入是 feats
            device_type='cuda',
            device_id=0,
            element_type=np.float32,
            shape=tuple(feats.shape),
            buffer_ptr=feats.data_ptr()
        )

        # 绑定输入 feats_length
        self.io_binding.bind_input(
            name=self.input_names[1],  # 假设第二个输入是 feats_length
            device_type='cuda',
            device_id=0,
            element_type=np.int32,
            shape=tuple(feats_length.shape),
            buffer_ptr=feats_length.data_ptr()
        )

        # 动态分配输出 Tensor（在 CUDA 上）
        output_tensor = torch.empty((1, feats.shape[-1] // 4), dtype=torch.int32, device=self.device).contiguous()
        self.io_binding.bind_output(
            name='indices',
            device_type='cuda',
            device_id=0,
            element_type=np.int32,
            shape=tuple(output_tensor.shape),
            buffer_ptr=output_tensor.data_ptr(),
        )

        # 执行推理
        self.session.run_with_iobinding(self.io_binding)

        # 输出 Tensor 已在 CUDA 上
        return output_tensor


class FeatureExtractor(nn.Module):
    """Base class for feature extractors."""

    def forward(self, audio: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Extract features from the given audio.

        Args:
            audio (Tensor): Input audio waveform.

        Returns:
            Tensor: Extracted features of shape (B, C, L), where B is the batch size,
                    C denotes output features, and L is the sequence length.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")


class MelSpectrogramFeatures(FeatureExtractor):
    def __init__(self, sample_rate=24000, n_fft=1024, hop_length=256, n_mels=100, padding="center"):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=padding == "center",
            power=1,
        )

    def forward(self, audio, **kwargs):
        if self.padding == "same":
            pad = self.mel_spec.win_length - self.mel_spec.hop_length
            audio = torch.nn.functional.pad(audio, (pad // 2, pad // 2), mode="reflect")
        mel = self.mel_spec(audio)
        features = safe_log(mel)
        return features


class EncodecFeatures(FeatureExtractor):
    def __init__(
            self,
            encodec_model: str = "encodec_24khz",
            bandwidths: List[float] = [1.5, 3.0, 6.0, 12.0],
            train_codebooks: bool = False,
    ):
        super().__init__()
        if encodec_model == "encodec_24khz":
            encodec = EncodecModel.encodec_model_24khz
        elif encodec_model == "encodec_48khz":
            encodec = EncodecModel.encodec_model_48khz
        else:
            raise ValueError(
                f"Unsupported encodec_model: {encodec_model}. Supported options are 'encodec_24khz' and 'encodec_48khz'."
            )
        self.encodec = encodec(pretrained=True)
        for param in self.encodec.parameters():
            param.requires_grad = False
        self.num_q = self.encodec.quantizer.get_num_quantizers_for_bandwidth(
            self.encodec.frame_rate, bandwidth=max(bandwidths)
        )
        codebook_weights = torch.cat([vq.codebook for vq in self.encodec.quantizer.vq.layers[: self.num_q]], dim=0)
        self.codebook_weights = torch.nn.Parameter(codebook_weights, requires_grad=train_codebooks)
        self.bandwidths = bandwidths

    @torch.no_grad()
    def get_encodec_codes(self, audio):
        audio = audio.unsqueeze(1)
        emb = self.encodec.encoder(audio)
        codes = self.encodec.quantizer.encode(emb, self.encodec.frame_rate, self.encodec.bandwidth)
        return codes

    def forward(self, audio: torch.Tensor, **kwargs):
        bandwidth_id = kwargs.get("bandwidth_id")
        if bandwidth_id is None:
            raise ValueError("The 'bandwidth_id' argument is required")
        self.encodec.eval()  # Force eval mode as Pytorch Lightning automatically sets child modules to training mode
        self.encodec.set_target_bandwidth(self.bandwidths[bandwidth_id])
        codes = self.get_encodec_codes(audio)
        # Instead of summing in the loop, it stores subsequent VQ dictionaries in a single `self.codebook_weights`
        # with offsets given by the number of bins, and finally summed in a vectorized operation.
        offsets = torch.arange(
            0, self.encodec.quantizer.bins * len(codes), self.encodec.quantizer.bins, device=audio.device
        )
        embeddings_idxs = codes + offsets.view(-1, 1, 1)
        features = torch.nn.functional.embedding(embeddings_idxs, self.codebook_weights).sum(dim=0)
        return features.transpose(1, 2)


class SnacFeatures(FeatureExtractor):
    def __init__(
            self,
            snac_model: str = "hubertsiuzdak/snac_24khz",
    ):
        super().__init__()

        self.snac_model = SNAC.from_pretrained(snac_model).eval()
        for param in self.snac_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def get_snac_features(self, audio_data):
        audio_data = audio_data[:, None, :]
        audio_data = self.snac_model.preprocess(audio_data)
        z = self.snac_model.encoder(audio_data)
        z_q, codes = self.snac_model.quantizer(z)
        return z_q

    def forward(self, audio: torch.Tensor, **kwargs):
        self.snac_model.eval()  # Force eval mode as Pytorch Lightning automatically sets child modules to training mode
        features = self.get_snac_features(audio)
        return features


class CosyvoiceFeatures(FeatureExtractor):
    def __init__(
            self,
            onnx_model: str = "/home/zhou/data3/tts/CosyVoice/pretrained_models/CosyVoice2-0.5B/speech_tokenizer_v2.onnx",
    ):
        super().__init__()
        self.tokenizer_model = None
        self.codebook = nn.Embedding(6561, 768)
        self.onnx_model = onnx_model

    @torch.no_grad()
    def get_codes(self, audio_data):
        if self.tokenizer_model is None:
            self.tokenizer_model = ONNXMultiInputDynamicWrapper(self.onnx_model)
        audio_data = torchaudio.functional.resample(audio_data, 24000, 16000)
        feats = whisper.log_mel_spectrogram(audio_data, n_mels=128)
        feats_length = np.array([feats.shape[2]], dtype=np.int32)
        feats_length = torch.from_numpy(feats_length).to(audio_data.device)
        codes = []
        for x in feats:
            y = self.tokenizer_model(x[None], feats_length)
            codes.append(y)
        codes = torch.cat(codes)
        return codes

    def forward(self, audio: torch.Tensor, **kwargs):
        codes = self.get_codes(audio)
        features = torch.nn.functional.embedding(codes, self.codebook.weight)
        return features.transpose(1, 2)


if __name__ == '__main__':
    m1 = EncodecFeatures()
    m2 = SnacFeatures()
    audio = torch.rand(1, 24000).to(torch.float32)
    print(m1(audio, bandwidth_id=0).shape)
    print(m2(audio).shape)
