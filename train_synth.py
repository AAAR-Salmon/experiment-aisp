#!/usr/bin/env python

import datetime
import json
import pickle
from typing import List

import numpy
import numpy.fft
import torch
import torch.utils.data

import models
import utils


def train_loop(dataloader, model, loss_fn, optimizer) -> None:
    model.train()
    size = len(dataloader.dataset)
    torch.autograd.set_detect_anomaly(True)

    for batch, (x, vq, y) in enumerate(dataloader):
        pred = model(x, vq)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def main(*, persons: List[str]):
    print("**** load data")
    waveforms = {}
    for person in persons:
        waveforms[person] = utils.load_waveforms(person)

    feature_voicequal = {}
    for person in persons:
        with open(f"{person}.np_fvq.pickle", "rb") as fp:
            feature_voicequal[person] = pickle.load(fp)

    print("**** make dataset")
    dataset_input_waveform = []
    dataset_input_voicequal = []
    dataset_output = []

    for waveforms_in in waveforms.values():
        for i in range(len(waveforms_in)):
            for person in persons:
                dataset_input_waveform.append(waveforms_in[i])
                dataset_input_voicequal.append(feature_voicequal[person])
                dataset_output.append(waveforms[person][i])

    dataset_input_waveform = torch.tensor(
        numpy.array(dataset_input_waveform, dtype=numpy.float32),
        dtype=torch.float32,
    )
    dataset_input_voicequal = torch.tensor(
        numpy.array(dataset_input_voicequal, dtype=numpy.float32),
        dtype=torch.float32,
    )
    dataset_output = torch.tensor(
        numpy.array(dataset_output, dtype=numpy.float32), dtype=torch.float32
    )

    train_dataset = torch.utils.data.TensorDataset(
        dataset_input_waveform, dataset_input_voicequal, dataset_output
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=4, shuffle=True
    )

    print("**** train")
    model = models.VoiceSynthesizer()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    epochs = 10
    for t in range(epochs):
        print(f"== epoch {t} =========")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        timestamp = datetime.datetime.utcnow().timestamp()
        torch.save(model.state_dict(), f"synth_state_dict-{timestamp}.pickle")


if __name__ == "__main__":
    with open("config.json", "r") as fp_config:
        config = json.load(fp_config)
    main(persons=config["persons"])
