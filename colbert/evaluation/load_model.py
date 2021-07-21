import os
import ujson
import torch
import random

from collections import defaultdict, OrderedDict

from transformers import AutoConfig
from colbert.parameters import DEVICE
from colbert.modeling.colbert import ColBERT
from colbert.utils.utils import print_message, load_checkpoint


def load_model(args, do_print=True):
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

    colbert = colbert.to(DEVICE)

    print_message("#> Loading model checkpoint.", condition=do_print)

    checkpoint = load_checkpoint(args.checkpoint, colbert, do_print=do_print)

    colbert.eval()

    return colbert, checkpoint
