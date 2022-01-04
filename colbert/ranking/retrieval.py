import os
import time
import faiss
import random
import torch
import itertools

from colbert.utils.runs import Run
from multiprocessing import Pool
from colbert.modeling.inference import ModelInference
from colbert.evaluation.ranking_logger import RankingLogger

from colbert.utils.utils import print_message, batch, read_titles
from colbert.ranking.rankers import Ranker
from colbert.parameters import DEVICE


def retrieve(args):
    inference = ModelInference(args.colbert, amp=args.amp)
    ranker = Ranker(args, inference, faiss_depth=args.faiss_depth)

    ranking_logger = RankingLogger(Run.path, qrels=None)
    milliseconds = 0
    titles = None
    if args.titles:
        titles = read_titles(args.titles)

    with ranking_logger.context(args.out_ranking_base,
                                also_save_annotations=False,
                                also_save_json=args.output_json,
                                ) as rlogger:
        queries = args.queries
        qids_in_order = list(queries.keys())

        for qoffset, qbatch in batch(qids_in_order, 100, provide_offset=True):
            qbatch_text = [queries[qid] for qid in qbatch]

            rankings = []

            for query_idx, q in enumerate(qbatch_text):
                if DEVICE.type == 'cuda' and args.nranks > 1:
                    torch.cuda.synchronize('cuda:0')
                s = time.time()

                Q = ranker.encode([q])
                pids, scores = ranker.rank(Q)

                if DEVICE.type == 'cuda' and args.nranks > 1:
                    torch.cuda.synchronize()
                milliseconds += (time.time() - s) * 1000.0

                if len(pids) & False: # TODO: temp get around, to add an option
                    print(qoffset+query_idx, q, len(scores), len(pids), scores[0], pids[0],
                          milliseconds / (qoffset+query_idx+1), 'ms')

                rankings.append(zip(pids, scores))

            for query_idx, (qid, ranking) in enumerate(zip(qbatch, rankings)):
                query_idx = qoffset + query_idx

                if query_idx % 100 == 0:
                    print_message(f"#> Logging query #{query_idx} (qid {qid}) now...")

                ranking = [(score, pid, None) for pid, score in itertools.islice(ranking, args.depth)]
                ranking = ranking[:args.top_n]
                rlogger.log(qid, ranking, is_ranked=True, queries=queries, titles=titles)

        if args.output_json:
            rlogger.log_json()

    print('\n\n')
    print(ranking_logger.filename)
    print(ranking_logger.filename + ".json\n")
    print("#> Done.")
    print('\n\n')
