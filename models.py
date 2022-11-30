import torch
import torch.nn

# 入力音声は WAV 形式
# - 16bit 符号あり整数
# - サンプリングレートは 16_000 Hz
# - 音声の時間は 10 sec = 160_000 frames
# - 1チャンネルモノラル
N_FRAMES = 160_000
# 片側化 FFT 後の要素数
N_RFFT = 80_001
# 声質特徴量の次元
N_DIM_VOICE_QUAL = 200
# 合成後 FIR フィルタ（ただし因果的でない）の次数
N_KERNEL_SIZE_FIR = 801
# 合成の中間層の次元
N_DIM_HIDDEN_SYNTH = 1_600


class VoiceQualEncoder(torch.nn.Module):
    def __init__(self, n_classes) -> None:
        super().__init__()

        self.n_classes = n_classes

        self._relu = torch.nn.ReLU()

        self.encode = torch.nn.Linear(N_RFFT, N_DIM_VOICE_QUAL)
        self._classify = torch.nn.Linear(N_DIM_VOICE_QUAL, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encode(x)
        x = self._classify(x)
        return x


class VoiceSynthesizer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._relu = torch.nn.ReLU()
        self._sigmoid = torch.nn.Sigmoid()

        self.synthesize1 = torch.nn.Linear(
            N_DIM_VOICE_QUAL + N_FRAMES, N_DIM_HIDDEN_SYNTH
        )
        self.synthesize2 = torch.nn.Linear(
            N_DIM_HIDDEN_SYNTH, N_FRAMES
        )
        self.fir_filter = torch.nn.Conv1d(
            1, 1, N_KERNEL_SIZE_FIR, bias=False
        )

    def forward(self, waveforms: torch.Tensor, vq: torch.Tensor) -> torch.Tensor:
        x = torch.concat([waveforms, vq], dim=1)
        x = self.synthesize1(x)
        x = self._relu(x)
        x = self.synthesize2(x)
        x = self._sigmoid(x)
        x = (x - 0.5) * 2  # transform value range (0, 1) -> (-1, 1)
        x = torch.concat([torch.zeros((x.shape[0], 800), dtype=x.dtype), x], dim=1)
        x = x.reshape(-1, 1, N_FRAMES + 800)
        x = self.fir_filter(x)
        return x
