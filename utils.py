import wave

import numpy as np


def load_waveforms(dirname: str):
    result = np.zeros((100, 160_000), np.float32)
    for i in range(100):
        path = f"{dirname}/{i+1:03d}.wav"
        with wave.open(path, "rb") as fw:
            nframes = fw.getnframes()
            signal = (
                np.frombuffer(fw.readframes(nframes), dtype=np.int16) + 0.5
            ) / 32767.5
            result[i, :nframes] = signal

    return result
