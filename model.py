import torch
import torch.nn

# 入力音声は WAV 形式
# - 16bit 符号あり整数
# - サンプリングレートは 16_000 Hz
# - 音声の時間は 10 sec = 160_000 frames
# - 1チャンネルモノラル
N_FRAMES = 160_000
# 声質特徴量の次元
N_DIM_VOICE_QUAL = 200
# 言語情報特徴量の次元
N_DIM_LANGUAGE = 1_600
# 合成後 FIR フィルタ（ただし因果的でない）の次数
N_KERNEL_SIZE_FIR = 801


class VoiceQualAutoEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.encode = torch.nn.Linear(N_FRAMES, N_DIM_VOICE_QUAL)
        self.decode = torch.nn.Linear(N_DIM_VOICE_QUAL, N_FRAMES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encode(x)
        x = self.decode(x)
        return x


class LanguageAutoEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.encode = torch.nn.Linear(N_FRAMES, N_DIM_LANGUAGE)
        self.decode = torch.nn.Linear(N_DIM_LANGUAGE, N_FRAMES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encode(x)
        x = self.decode(x)
        return x


class VoiceSynthesizer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.synthesize = torch.nn.Linear(
            N_DIM_VOICE_QUAL + N_DIM_LANGUAGE, N_FRAMES
        )
        self.fir_filter = torch.nn.Conv1d(
            1, 1, N_KERNEL_SIZE_FIR, bias=False, padding="zero"
        )

    def forward(self, vq: torch.Tensor, lang: torch.Tensor) -> torch.Tensor:
        x = torch.concat([vq, lang])
        x = self.synthesize(x)
        x = self.fir_filter(x)
        return x
