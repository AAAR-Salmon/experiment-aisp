#!/usr/bin/env python

import json
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

    for batch, (x, y) in enumerate(dataloader):
        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    model.eval()
    num_batches = len(dataloader)
    test_loss = 0

    for x, y in dataloader:
        pred = model(x)
        test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")


def main(*, persons: List[str]):
    print("**** load data")
    waveforms = {}
    for person in persons:
        waveforms[person] = utils.load_waveforms(person)

    print("**** calc logarithm of power spectrums")
    powerspecs = {
        person: numpy.log10(
            numpy.abs(numpy.fft.rfft(waveforms_of_person, norm="forward"))
        )
        for person, waveforms_of_person in waveforms.items()
    }

    print("**** make dataset")
    dataset_input = numpy.vstack(list(powerspecs.values()))
    indices = numpy.arange(len(dataset_input))
    numpy.random.shuffle(indices)
    dataset_output = dataset_input[indices]

    dataset_input = torch.tensor(dataset_input, dtype=torch.float32)
    dataset_output = torch.tensor(dataset_output, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(dataset_input, dataset_output)
    train_dataset_length = int(0.8 * len(dataset))
    test_dataset_length = len(dataset) - train_dataset_length
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, lengths=[train_dataset_length, test_dataset_length]
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=4, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=False)

    print("**** train")
    model = models.VoiceQualAutoEncoder()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    epochs = 10
    for t in range(epochs):
        print(f"== epoch {t} =========")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)


if __name__ == "__main__":
    with open("config.json", "r") as fp_config:
        config = json.load(fp_config)
    main(persons=config["persons"])
