from vocos import CosyvoiceVocos
vocoder = CosyvoiceVocos.from_pretrained('/home/zhou/data3/tts/vocos/logs/lightning_logs/version_13').eval()
vocoder.export_to_onnx('test.onnx')