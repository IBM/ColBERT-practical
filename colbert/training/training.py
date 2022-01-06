import os
import math
import random
import time
import torch
import torch.nn as nn
import numpy as np

from transformers import AdamW, AutoConfig
from colbert.utils.runs import Run
from colbert.utils.amp import MixedPrecisionManager

from colbert.training.lazy_batcher import LazyBatcher
# from colbert.training.eager_batcher import EagerBatcher
from colbert.training.eager_batcher_2 import EagerBatcher
from colbert.parameters import DEVICE

from colbert.modeling.colbert import ColBERT
from colbert.utils.utils import print_message
from colbert.training.utils import print_progress, manage_checkpoints
from utility.utilities import rel_link_last_file


def train(args):
    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)
    if args.distributed:
        torch.cuda.manual_seed_all(12345)

    if args.distributed:
        assert args.bsize % args.nranks == 0, (args.bsize, args.nranks)
        assert args.accumsteps == 1
        args.bsize = args.bsize // args.nranks

        print("Using args.bsize =", args.bsize, "(per process) and args.accumsteps =", args.accumsteps)

    if args.lazy:
        reader = LazyBatcher(args, (0 if args.rank == -1 else args.rank), args.nranks)
    else:
        reader = EagerBatcher(args, (0 if args.rank == -1 else args.rank), args.nranks)

    if args.rank not in [-1, 0]:
        torch.distributed.barrier()

    config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=args.model_name,
    )
    colbert = ColBERT.from_pretrained(
        pretrained_model_name_or_path=args.model_name,
        config=config,
        query_maxlen=args.query_maxlen,
        doc_maxlen=args.doc_maxlen,
        dim=args.dim,
        similarity_metric=args.similarity,
        mask_punctuation=args.mask_punctuation,
    )

    if args.pretrained_model is not None:
        pretrained_model = torch.load(args.pretrained_model, map_location='cpu')

        if 'model_state_dict' in pretrained_model:
            pretrained_model = pretrained_model['model_state_dict']

        try:
            colbert.load_state_dict(pretrained_model)
        except:
            print_message("[WARNING] Loading pretrained_model with strict=False")
            colbert.load_state_dict(pretrained_model, strict=False)

    if args.checkpoint is not None:
        assert args.resume_optimizer is False, "TODO: This would mean reload optimizer too."
        print_message(f"#> Starting from checkpoint {args.checkpoint} -- but NOT the optimizer!")

        checkpoint = torch.load(args.checkpoint, map_location='cpu')

        try:
            colbert.load_state_dict(checkpoint['model_state_dict'])
        except:
            print_message("[WARNING] Loading checkpoint with strict=False")
            colbert.load_state_dict(checkpoint['model_state_dict'], strict=False)

    if args.rank == 0:
        torch.distributed.barrier()

    colbert = colbert.to(DEVICE)
    colbert.train()

    if args.distributed:
        colbert = torch.nn.parallel.DistributedDataParallel(colbert, device_ids=[args.rank],
                                                            output_device=args.rank,
                                                            find_unused_parameters=True)

    optimizer = AdamW(filter(lambda p: p.requires_grad, colbert.parameters()), lr=args.lr, eps=1e-8)
    optimizer.zero_grad()

    amp = MixedPrecisionManager(args.amp)
    criterion = nn.CrossEntropyLoss()
    labels = torch.zeros(args.bsize, dtype=torch.long, device=DEVICE)

    start_time = time.time()
    train_loss = 0.0

    start_batch_idx = 0

    if args.resume:
        assert args.checkpoint is not None
        start_batch_idx = checkpoint['batch']

        reader.skip_to_batch(start_batch_idx, checkpoint['arguments']['bsize'])

    maxsteps = min(args.maxsteps, math.ceil((args.epochs * len(reader)) / (args.bsize * args.nranks)))
    path = os.path.join(Run.path, 'checkpoints')
    Run.info("maxsteps: {}".format(args.maxsteps))
    Run.info("{} epochs of {} examples".format(args.epochs, len(reader)))
    Run.info("batch size: {}".format(args.bsize))
    Run.info("maxsteps set to {}".format(maxsteps))
    print_every_step = False
    for batch_idx, BatchSteps in zip(range(start_batch_idx, maxsteps), reader):
        n_instances = batch_idx * args.bsize * args.nranks
        if (n_instances + 1) % len(reader) < args.bsize * args.nranks:
            Run.info("====== Epoch {}...".format((n_instances+1) // len(reader)))
            reader.shuffle()
        # Run.info("Batch {}".format(batch_idx))
        if batch_idx % 100 == 0:
            Run.info("Batch {}".format(batch_idx))
        this_batch_loss = 0.0
        for queries, passages in BatchSteps:
            with amp.context():
                scores = colbert(queries, passages).view(2, -1).permute(1, 0)
                loss = criterion(scores, labels[:scores.size(0)])
                loss = loss / args.accumsteps

            if args.rank < 1 and print_every_step:
                print_progress(scores)

            amp.backward(loss)

            train_loss += loss.item()
            this_batch_loss += loss.item()

        amp.step(colbert, optimizer)

        if args.rank < 1:
            avg_loss = train_loss / (batch_idx+1)

            num_examples_seen = (batch_idx - start_batch_idx) * args.bsize * args.nranks
            elapsed = float(time.time() - start_time)

            log_to_mlflow = (batch_idx % 20 == 0)
            Run.log_metric('train/avg_loss', avg_loss, step=batch_idx, log_to_mlflow=log_to_mlflow)
            Run.log_metric('train/batch_loss', this_batch_loss, step=batch_idx, log_to_mlflow=log_to_mlflow)
            Run.log_metric('train/examples', num_examples_seen, step=batch_idx, log_to_mlflow=log_to_mlflow)
            Run.log_metric('train/throughput', num_examples_seen / elapsed, step=batch_idx, log_to_mlflow=log_to_mlflow)

            # print_message(batch_idx, avg_loss)
            num_per_epoch = len(reader)
            epoch_idx = ((batch_idx+1) * args.bsize * args.nranks) // num_per_epoch - 1
            manage_checkpoints(args, colbert, optimizer, batch_idx+1, num_per_epoch, epoch_idx)

    rel_link_last_file(path, "colbert-LAST.dnn", "*.model")

