#!/usr/bin/env python

import json
import pickle
import sys
import wave
from typing import List

import numpy as np
import numpy.typing
import torch

import models


def main(
    persons: List[str],
    input_waveform: numpy.typing.NDArray[np.float32],
    fp_state_dict,
):
    model = models.VoiceSynthesizer()
    model.load_state_dict(torch.load(fp_state_dict))

    feature_voicequal = {}
    for person in persons:
        with open(f"{person}.np_fvq.pickle", "rb") as fp:
            feature_voicequal[person] = torch.tensor(
                pickle.load(fp), dtype=torch.float32
            ).reshape(1, -1)

    for person, feature_voicequal_of_person in feature_voicequal.items():
        output_waveform: numpy.typing.NDArray[np.float32] = (
            model(input_waveform, feature_voicequal_of_person)
            .reshape(-1)
            .detach()
            .numpy()
        )
        output_waveform = (output_waveform * 32767).astype(np.int16)
        with wave.open(f"output-{person}.wav", "wb") as fp_wav_output:
            fp_wav_output.setnchannels(1)
            fp_wav_output.setframerate(16000)
            fp_wav_output.setnframes(160000)
            fp_wav_output.setsampwidth(2)
            fp_wav_output.writeframes(output_waveform.tobytes())


if __name__ == "__main__":
    path_state_dict, path_input = sys.argv[1:]
    with (
        open(path_state_dict, "rb") as fp_state_dict,
        wave.open(path_input, "rb") as wav_input,
    ):
        input_waveform = np.zeros(160_000, np.float32)
        n_frames = wav_input.getnframes()
        signal = np.frombuffer(
            wav_input.readframes(n_frames), dtype=np.int16
        ).astype(np.float32)
        signal = (signal + 0.5) / 32767.5
        input_waveform[:n_frames] = signal
        input_waveform = torch.tensor(input_waveform, dtype=torch.float32).reshape(1, -1)

        with open("config.json", "r") as fp_config:
            config = json.load(fp_config)
        main(config["persons"], input_waveform, fp_state_dict)
