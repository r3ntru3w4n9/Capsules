import json
import os
from argparse import ArgumentParser
from collections import defaultdict
from os import path as os_path

import torch
from matplotlib import pyplot as plt
from torch import autograd, cuda
from torch import multiprocessing as mp
from torch import optim
from torch.nn import functional as F
from torch.nn import utils as nn_utils
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

import functions
from datasets import CacheDataset, DelegateDataset
from losses import MarginLoss, ReconLoss
from networks import CapsDecoder, CapsNet, Cequential, FeatureExtractor

if __name__ == "__main__":
    parser = ArgumentParser()

    convolution = parser.add_argument_group("convolution")
    convolution.add_argument(
        "-Cc", "--conv-channels", type=int, nargs="+", default=(1, 256, 32)
    )
    convolution.add_argument(
        "-Ck", "--conv-kernels", type=int, nargs="+", default=(9, 9)
    )
    convolution.add_argument(
        "-Cs", "--conv-strides", type=int, nargs="+", default=(1, 2)
    )

    capsule = parser.add_argument_group("capsule")
    capsule.add_argument(
        "-ch", "--caps-channels", type=int, nargs="+", default=(32 * 6 * 6, 16)
    )
    capsule.add_argument("-cc", "--caps-capsules", type=int, nargs="+", default=(8, 10))
    capsule.add_argument("-ci", "--caps-iters", type=int, default=3)

    decoder = parser.add_argument_group("decoder")
    decoder.add_argument(
        "-dl",
        "--decoder-layers",
        type=int,
        nargs="+",
        default=(16 * 10, 512, 1024, 784),
    )

    augmentation = parser.add_argument_group("augmentation")
    augmentation.add_argument("-am", "--aug-mean", type=float, default=None)
    augmentation.add_argument("-as", "--aug-std", type=float, default=None)
    augmentation.add_argument("-ar", "--aug-rotation", type=float, default=0)
    augmentation.add_argument("-ab", "--aug-brightness", type=float, default=0)
    augmentation.add_argument("-ac", "--aug-contrast", type=float, default=0)
    augmentation.add_argument("-at", "--aug-saturation", type=float, default=0)

    losses = parser.add_argument_group("losses")
    losses.add_argument("-lt", "--Tc", type=float, default=1)
    losses.add_argument("-ll", "--lmbda", type=float, default=0.5)
    losses.add_argument("-lb", "--bound", type=float, nargs=2, default=(1, 0.1))
    losses.add_argument("-lw", "--recon-weight", type=float, default=1)

    parser.add_argument("-e", "--epochs", type=int, default=100)
    parser.add_argument("-b", "--batch", type=int, default=1000)
    parser.add_argument("-lr", "--learning-rate", type=float, default=1e-3)
    parser.add_argument("-mg", "--max-grad", type=float, default=None)
    parser.add_argument("-d", "--device", type=str, default="cuda")
    parser.add_argument("-P", "--processes", type=int, default=0)
    parser.add_argument("-r", "--dataset-root", type=str, default=".")
    parser.add_argument("-s", "--save", type=int, default=None)
    parser.add_argument("-o", "--out", type=str, default="")
    parser.add_argument("-p", "--plot", action="store_true")
    parser.add_argument("-tnan", "--terminate-on-nan", action="store_true")
    parser.add_argument("-pm", "--print-module", action="store_true")

    flags = parser.parse_args()

    mp.set_start_method(method="spawn")

    conv_channels = flags.conv_channels
    conv_kernels = flags.conv_kernels
    conv_strides = flags.conv_strides

    caps_channels = flags.caps_channels
    caps_capsules = flags.caps_capsules
    caps_iters = flags.caps_iters

    decoder_layers = flags.decoder_layers

    losses_Tc = flags.Tc
    losses_lmbda = flags.lmbda
    losses_bound = flags.bound
    losses_weight = flags.recon_weight

    aug_mean = flags.aug_mean
    aug_std = flags.aug_std
    aug_rotation = flags.aug_rotation
    aug_brightness = flags.aug_rotation
    aug_contrast = flags.aug_contrast
    aug_saturation = flags.aug_saturation

    epochs = flags.epochs
    batch = flags.batch
    learning_rate = flags.learning_rate
    max_grad = flags.max_grad
    device = flags.device if cuda.is_available() else "cpu"
    processes = flags.processes
    dataset_root = flags.dataset_root
    save = flags.save
    out = flags.out
    plot = flags.plot
    terminate_on_nan = flags.terminate_on_nan
    print_module = flags.print_module

    if out:
        os.makedirs(name=out, exist_ok=True)

    transform_list = []

    if any((aug_brightness, aug_contrast, aug_saturation)):
        transform_list.append(
            transforms.ColorJitter(
                brightness=aug_brightness,
                contrast=aug_contrast,
                saturation=aug_saturation,
            )
        )

    if aug_rotation:
        transform_list.append(transforms.RandomRotation(degrees=aug_rotation))

    transform_list.append(transforms.ToTensor())

    if all(((aug_mean is not None), (aug_std is not None))):
        transform_list.append(transforms.Normalize(mean=(aug_mean,), std=(aug_std,)))

    data_transform = transforms.Compose(transform_list)

    train_dataset = datasets.MNIST(
        root=dataset_root, train=True, download=True, transform=data_transform
    )
    eval_dataset = datasets.MNIST(
        root=dataset_root, train=False, download=True, transform=transforms.ToTensor()
    )

    categories = len(train_dataset.classes)
    assert categories == len(eval_dataset.classes)

    if len(transform_list) == 1:
        train_dataset = CacheDataset(dataset=train_dataset, device=device)
        eval_dataset = CacheDataset(dataset=eval_dataset, device=device)
    else:
        train_dataset = DelegateDataset(dataset=train_dataset, device=device)
        eval_dataset = DelegateDataset(dataset=eval_dataset, device=device)

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch, shuffle=True, num_workers=processes
    )
    eval_loader = DataLoader(
        dataset=eval_dataset, batch_size=batch, shuffle=False, num_workers=processes
    )

    capsnet = CapsNet(
        feature_net=FeatureExtractor(
            conv_channels=conv_channels,
            kernel_sizes=conv_kernels,
            strides=conv_strides,
            out_features=caps_capsules[0],
        ),
        capsule_net=Cequential(
            channels=caps_channels, capsules=caps_capsules, num_iters=caps_iters
        ),
        decoder_net=CapsDecoder(layers=decoder_layers, categories=categories),
    ).to(device)
    optimizer = optim.Adam(params=capsnet.parameters(), lr=learning_rate)

    if print_module:
        print(capsnet)

    if terminate_on_nan:
        capsnet.apply(
            fn=lambda module: module.register_backward_hook(
                hook=functions.terminate_on_nan
            )
        )

    (margin_loss, recon_loss) = (
        MarginLoss(
            T_c=losses_Tc,
            lmbda=losses_lmbda,
            boundary=losses_bound,
            categories=categories,
        ),
        ReconLoss(),
    )

    history = defaultdict(list)
    for epoch in range(1, 1 + epochs):
        print(f"epoch: {epoch:04d}/{epochs:04d}, training")
        capsnet.train()
        (correct, total) = (0, 0)
        (total_margin_loss, total_recon_loss) = (0, 0)
        for (images, labels) in tqdm(train_loader):
            (predict, recon) = capsnet(images, labels)
            mar_loss = margin_loss(predict, labels)
            rec_loss = losses_weight * recon_loss(recon, images)
            loss = mar_loss + rec_loss
            optimizer.zero_grad()
            loss.backward()
            if max_grad:
                nn_utils.clip_grad_norm_(
                    parameters=capsnet.parameters(), max_norm=max_grad
                )
            optimizer.step()

            correct += ((predict ** 2).sum(dim=1).argmax(dim=-1) == labels).sum().item()
            total_margin_loss += mar_loss.item()
            total_recon_loss += rec_loss.item()
            total += len(labels)

        history["train_accuracy"].append(correct / total)
        history["train_margin_loss"].append(total_margin_loss)
        history["train_recon_loss"].append(total_recon_loss)

        print(
            f"accuracy: {correct/total:4.3f}, "
            f"margin loss: {total_margin_loss:08.4f} "
            f"recon loss: {total_recon_loss:08.4f}"
        )

        print(f"epoch: {epoch:04d}/{epochs:04d}, testing")
        capsnet.eval()
        with torch.no_grad():
            (correct, total) = (0, 0)
            (total_margin_loss, total_recon_loss) = (0, 0)
            for (images, labels) in tqdm(eval_loader):
                (predict, recon) = capsnet(images, labels)
                correct += (
                    ((predict ** 2).sum(dim=1).argmax(dim=-1) == labels).sum().item()
                )
                total += len(labels)
                mar_loss = margin_loss(predict, labels)
                rec_loss = recon_loss(recon, images)
                total_margin_loss += mar_loss.item()
                total_recon_loss += rec_loss.item()

            history["eval_accuracy"].append(correct / total)
            history["eval_margin_loss"].append(total_margin_loss)
            history["eval_recon_loss"].append(total_recon_loss)

            print(
                f"accuracy: {correct/total:4.3f}, "
                f"margin loss: {total_margin_loss:08.4f} "
                f"recon loss: {total_recon_loss:08.4f}"
            )

            if out:
                if (epoch % save) == 0:
                    torch.save(
                        obj=capsnet.state_dict(), f=os_path.join(out, f"{epoch:04d}.pt")
                    )

    if out:
        json.dump(
            obj=history, fp=open(file=os_path.join(out, "history.json"), mode="w+")
        )
        if plot:
            for (name, plot) in history.items():
                plt.clf()
                plt.plot(plot)
                plt.savefig(os_path.join(out, f"{name}.png"))
