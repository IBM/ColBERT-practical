import os
import math
import torch

from colbert.utils.runs import Run
from colbert.utils.utils import print_message, save_checkpoint
from colbert.parameters import SAVED_CHECKPOINTS


def print_progress(scores):
    positive_avg, negative_avg = round(scores[:, 0].mean().item(), 2), round(scores[:, 1].mean().item(), 2)
    print("#>>>   ", positive_avg, negative_avg, '\t\t|\t\t', positive_avg - negative_avg)


def manage_checkpoints(args, colbert, optimizer, batch_idx, num_per_epoch, epoch_idx=0):
    arguments = args.input_arguments.__dict__

    saved_name = ""
    path = os.path.join(Run.path, 'checkpoints')

    if not os.path.exists(path):
        os.mkdir(path)
    prefix = os.path.join(path, "colbert.dnn")

    if args.save_epochs == -1:
        if batch_idx % args.save_steps == 0:
            saved_name = prefix + ".epoch_{}_batch_{}.model".format(0, batch_idx)
            save_checkpoint(saved_name, epoch_idx, batch_idx, colbert, optimizer, arguments)
    else:
        if batch_idx * args.bsize * args.nranks % int(args.save_epochs * num_per_epoch) < args.bsize * args.nranks:
            if args.save_epochs.is_integer():
                saved_name = prefix + ".epoch_{}_batch_{}.model".format(epoch_idx, 0)
            else:
                saved_name = prefix + ".epoch_{}_batch_{}.model".format(epoch_idx, batch_idx)

            save_checkpoint(saved_name, epoch_idx, batch_idx, colbert, optimizer, arguments)

    if batch_idx in SAVED_CHECKPOINTS:
        name = prefix + ".epoch_{}_batch_{}.model".format(0, batch_idx)
        if not name == saved_name:
            save_checkpoint(name, epoch_idx, batch_idx, colbert, optimizer, arguments)

    if (batch_idx * args.bsize * args.nranks) % (args.epochs * num_per_epoch) < args.bsize * args.nranks:
        name = prefix + ".epoch_{}_batch_{}.model".format(args.epochs - 1, 0)
        if not name == saved_name:
            save_checkpoint(name, epoch_idx, batch_idx, colbert, optimizer, arguments)
