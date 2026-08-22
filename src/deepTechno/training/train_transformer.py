"""Music Transformer training loop (PyTorch)."""
import csv
import os
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from deepTechno.data.dataset import MaestroDataset, compute_epiano_accuracy
from deepTechno.model.constants import (
    ADAM_BETA_1, ADAM_BETA_2, ADAM_EPSILON,
    LR_DEFAULT_START, SCHEDULER_WARMUP_STEPS,
    TOKEN_PAD, VOCAB_SIZE, SEPERATOR,
)
from deepTechno.model.loss import SmoothCrossEntropyLoss
from deepTechno.model.transformer import MusicTransformer
from deepTechno.utils.device import get_device, use_cuda
from deepTechno.utils.lr_scheduling import LrStepTracker, get_lr


@dataclass
class TransformerConfig:
    train_csv: str = "dataset/train.csv"
    val_csv:   str = "dataset/val.csv"
    output_dir: str = "./results/transformer"
    n_layers: int = 6
    num_heads: int = 8
    d_model: int = 512
    dim_feedforward: int = 1024
    dropout: float = 0.1
    max_sequence: int = 2048
    rpr: bool = True
    batch_size: int = 2
    epochs: int = 100
    lr: float = None          # None → use warmup schedule
    ce_smoothing: float = None
    weight_modulus: int = 10
    print_modulus: int = 1
    n_workers: int = 1
    force_cpu: bool = False
    no_tensorboard: bool = False
    continue_weights: str = None
    continue_epoch: int = None


CSV_HEADER = ["Epoch", "Learn rate", "Avg Train loss", "Train Accuracy", "Avg Eval loss", "Eval Accuracy"]
BASELINE_EPOCH = -1


def train_epoch(cur_epoch, model, dataloader, loss_fn, opt, lr_scheduler=None, print_modulus=1):
    model.train()
    for batch_num, (x, tgt) in enumerate(dataloader):
        t0 = time.time()
        opt.zero_grad()
        x, tgt = x.to(get_device()), tgt.to(get_device())
        y = model(x)
        y   = y.reshape(y.shape[0] * y.shape[1], -1)
        tgt = tgt.flatten()
        loss = loss_fn(y, tgt)
        loss.backward()
        opt.step()
        if lr_scheduler:
            lr_scheduler.step()
        if (batch_num + 1) % print_modulus == 0:
            print(f"{SEPERATOR}\nEpoch {cur_epoch} Batch {batch_num+1}/{len(dataloader)}")
            print(f"LR: {get_lr(opt):.6f}  Train loss: {float(loss):.4f}  Time: {time.time()-t0:.1f}s")


def eval_model(model, dataloader, loss_fn):
    model.eval()
    sum_loss = sum_acc = 0.0
    n = len(dataloader)
    with torch.no_grad():
        for x, tgt in dataloader:
            x, tgt = x.to(get_device()), tgt.to(get_device())
            y = model(x)
            sum_acc += float(compute_epiano_accuracy(y, tgt))
            y   = y.reshape(y.shape[0] * y.shape[1], -1)
            tgt = tgt.flatten()
            sum_loss += float(loss_fn(y, tgt))
    return sum_loss / n, sum_acc / n


def run_transformer_training(config: TransformerConfig):
    if config.force_cpu:
        use_cuda(False)

    os.makedirs(config.output_dir, exist_ok=True)
    weights_dir  = os.path.join(config.output_dir, "weights");  os.makedirs(weights_dir,  exist_ok=True)
    results_dir  = os.path.join(config.output_dir, "results");  os.makedirs(results_dir,  exist_ok=True)
    results_file = os.path.join(results_dir, "results.csv")
    best_loss_file = os.path.join(results_dir, "best_loss.pt")
    best_acc_file  = os.path.join(results_dir, "best_acc.pt")

    train_dataset = MaestroDataset(config.train_csv, config.max_sequence)
    val_dataset   = MaestroDataset(config.val_csv,   config.max_sequence)
    train_loader  = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=config.n_workers, shuffle=True)
    val_loader    = DataLoader(val_dataset,   batch_size=config.batch_size, num_workers=config.n_workers)

    model = MusicTransformer(
        n_layers=config.n_layers, num_heads=config.num_heads, d_model=config.d_model,
        dim_feedforward=config.dim_feedforward, dropout=config.dropout,
        max_sequence=config.max_sequence, rpr=config.rpr,
    ).to(get_device())

    start_epoch = BASELINE_EPOCH
    if config.continue_weights:
        model.load_state_dict(torch.load(config.continue_weights))
        start_epoch = config.continue_epoch

    eval_loss_fn  = nn.CrossEntropyLoss(ignore_index=TOKEN_PAD)
    train_loss_fn = (SmoothCrossEntropyLoss(config.ce_smoothing, VOCAB_SIZE, ignore_index=TOKEN_PAD)
                     if config.ce_smoothing else eval_loss_fn)

    lr = config.lr or LR_DEFAULT_START
    init_step = 0 if not config.continue_epoch else config.continue_epoch * len(train_loader)
    lr_stepper = LrStepTracker(config.d_model, SCHEDULER_WARMUP_STEPS, init_step)

    opt = Adam(model.parameters(), lr=lr, betas=(ADAM_BETA_1, ADAM_BETA_2), eps=ADAM_EPSILON)
    lr_scheduler = LambdaLR(opt, lr_stepper.step) if config.lr is None else None

    tensorboard = None
    if not config.no_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        tensorboard = SummaryWriter(log_dir=os.path.join(config.output_dir, "tensorboard"))

    if not os.path.isfile(results_file):
        with open(results_file, "w", newline="") as fh:
            csv.writer(fh).writerow(CSV_HEADER)

    best_eval_acc = 0.0
    best_eval_loss = float("inf")

    for epoch in range(start_epoch, config.epochs):
        if epoch > BASELINE_EPOCH:
            print(f"\n{SEPERATOR}\nEPOCH {epoch+1}\n{SEPERATOR}")
            train_epoch(epoch + 1, model, train_loader, train_loss_fn, opt, lr_scheduler, config.print_modulus)

        train_loss, train_acc = eval_model(model, train_loader, train_loss_fn)
        eval_loss,  eval_acc  = eval_model(model, val_loader,   eval_loss_fn)
        lr_now = get_lr(opt)

        print(f"Epoch {epoch+1} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | eval_loss={eval_loss:.4f} eval_acc={eval_acc:.4f}")

        if eval_acc > best_eval_acc:
            best_eval_acc = eval_acc
            torch.save(model.state_dict(), best_acc_file)
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            torch.save(model.state_dict(), best_loss_file)

        if tensorboard:
            tensorboard.add_scalar("Loss/train", train_loss, epoch + 1)
            tensorboard.add_scalar("Loss/eval",  eval_loss,  epoch + 1)
            tensorboard.add_scalar("Acc/train",  train_acc,  epoch + 1)
            tensorboard.add_scalar("Acc/eval",   eval_acc,   epoch + 1)
            tensorboard.add_scalar("LR",         lr_now,     epoch + 1)
            tensorboard.flush()

        if (epoch + 1) % config.weight_modulus == 0:
            torch.save(model.state_dict(), os.path.join(weights_dir, f"epoch_{epoch+1:04d}.pt"))

        with open(results_file, "a", newline="") as fh:
            csv.writer(fh).writerow([epoch + 1, lr_now, train_loss, train_acc, eval_loss, eval_acc])

    if tensorboard:
        tensorboard.flush()

    return model
