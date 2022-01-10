import os
import ujson
import random
import math
import time
from argparse import ArgumentParser

from colbert.utils.runs import Run
from colbert.utils.parser import Arguments
import colbert.utils.distributed as distributed
from self_training_w_gold import get_self_guided_training_w_gold

from colbert.utils.utils import print_message, create_directory
from colbert.indexing.encoder import CollectionEncoder
from colbert.indexing.faiss import index_faiss, get_faiss_index_name
from colbert.indexing.loaders import load_doclens
from colbert.evaluation.loaders import load_colbert, load_qrels, load_queries
from colbert.ranking.retrieval import retrieve
from colbert.ranking.batch_retrieval import batch_retrieve
from utility.utilities import rel_link_last_file, get_file_new_link_and_timestamp


def do_index(args):
    Run.info("==== indexing ====")

    process_idx = max(0, args.rank)
    encoder = CollectionEncoder(args, process_idx=process_idx, num_processes=args.nranks)
    encoder.encode()

    distributed.barrier(args.rank)

    # Save metadata.
    if args.rank < 1:
        metadata_path = os.path.join(args.index_path, 'metadata.json')
        print_message("Saving (the following) metadata to", metadata_path, "..")
        print(args.input_arguments)

        with open(metadata_path, 'w') as output_metadata:
            ujson.dump(args.input_arguments.__dict__, output_metadata)

    distributed.barrier(args.rank)


def do_faiss_index(args):
    # ========= faiss indexing
    Run.info("==== faiss indexing ====")
    assert os.path.exists(args.index_path)

    num_embeddings = sum(load_doclens(args.index_path))
    print("#> num_embeddings =", num_embeddings)

    if args.partitions is None:
        args.partitions = 1 << math.ceil(math.log2(8 * math.sqrt(num_embeddings)))
        print('\n\n')
        Run.warn("You did not specify --partitions!")
        Run.warn("Default computation chooses", args.partitions,
                 "partitions (for {} embeddings)".format(num_embeddings))
        print('\n\n')

    index_faiss(args)


def do_retrieve(args):
    # ========= retrieval
    Run.info("==== retrieval ====")

    args.colbert, args.checkpoint = load_colbert(args)
    args.qrels = load_qrels(args.qrels)
    args.queries = load_queries(args.queries_file)

    if args.faiss_name is not None:
        args.faiss_index_path = os.path.join(args.index_path, args.faiss_name)
    else:
        args.faiss_index_path = os.path.join(args.index_path, get_faiss_index_name(args))

    if args.batch:
        batch_retrieve(args)
    else:
        retrieve(args)


class AsynArguments(Arguments):
    def __init__(self, description):
        self.parser = ArgumentParser(description=description, conflict_handler="resolve")
        self.checks = []

        self.add_argument('--root', dest='root', default='experiments')
        self.add_argument('--experiment', dest='experiment', default='dirty')
        self.add_argument('--run', dest='run', default=Run.name)
        self.add_argument('--local_rank', dest='rank', default=-1, type=int)


def get_params():
    parser = AsynArguments(
        description='Precomputing document representations with ColBERT.'
    )

    # indexing params
    parser.add_model_parameters()
    parser.add_model_inference_parameters()
    parser.add_indexing_input()
    parser.add_argument('--chunksize', dest='chunksize', default=6.0, required=False, type=float)   # in GiBs

    # faiss indexing params
    parser.add_argument('--partitions', dest='partitions', default=None, type=int)
    parser.add_argument('--sample', dest='sample', default=None, type=float)
    parser.add_argument('--slices', dest='slices', default=1, type=int)

    # retrieval params
    parser.add_retrieval_input()
    parser.add_write_ranking_result_input()
    parser.add_argument('--queries', dest='queries_file', default=None)
    parser.add_argument('--collection', dest='collection', default=None)
    parser.add_argument('--qrels', dest='qrels', default=None)

    parser.add_argument('--faiss_name', dest='faiss_name', default=None, type=str)
    parser.add_argument('--faiss_depth', dest='faiss_depth', default=1024, type=int)
    parser.add_argument('--part-range', dest='part_range', default=None, type=str)
    parser.add_argument('--batch', dest='batch', default=False, action='store_true')
    parser.add_argument('--depth', dest='depth', default=1000, type=int)

    parser.add_argument('--n_rounds', dest='n_rounds', default=1, type=int,
                        help="How many rounds of loading training data and do epochs training, "
                             "used for asynchronous training.")

    parser.add_argument('--positive', dest='positive', required=True, type=str)
    parser.add_argument('--depth-', dest='depth_negative', required=True, type=int)
    parser.add_argument('--max_n_neg', dest='max_n_neg', required=False, default=100, type=int)
    parser.add_argument('--min_n_neg', dest='min_n_neg', required=False, default=3, type=int)
    parser.add_argument('--sample_strategy', dest='sample_strategy', required=False,
                        default='s1', type=str, choices=['s1', 's2'])
    # args for strategy 2
    parser.add_argument('--depth-e', dest='depth_easy_negative', required=False, default=1000, type=int)
    parser.add_argument('--n_neg_hard', dest='n_neg_hard', required=False, default=100, type=int)

    args = parser.parse()

    # backup args.checkpoint because in ColBERT model loading, args.checkpoint will be overwritten,
    # but we need to load it again in later rounds
    args.checkpoint_bak = args.checkpoint

    # checking for faiss indexing params
    assert args.slices >= 1
    assert args.sample is None or (0.0 < args.sample < 1.0), args.sample

    # checking for retrieval params
    args.depth = args.depth if args.depth > 0 else None
    if args.part_range:
        part_offset, part_endpos = map(int, args.part_range.split('..'))
        args.part_range = range(part_offset, part_endpos)

    return args


def main():
    random.seed(12345)
    args = get_params()

    with Run.context():
        prev_model_time = "--"
        start_time_0 = time.time()
        for idx_round in range(args.n_rounds):

            start_time = time.time()
            args.index_path = os.path.join(args.index_root,
                                           args.index_name,
                                           f"round_{idx_round}",
                                           )

            distributed.barrier(args.rank)

            if args.rank < 1:
                # create_directory(args.index_root)
                create_directory(args.index_path)

            distributed.barrier(args.rank)

            args.checkpoint = args.checkpoint_bak

            # make sure we load a new model for every round
            model_link, model_time = get_file_new_link_and_timestamp(args.checkpoint, prev_model_time)
            Run.info(f"Model time: {model_time} {model_link}")
            prev_model_time = model_time

            real_start_time = time.time()
            do_index(args)

            do_faiss_index(args)

            args.out_ranking_base = f"train.ranking.round_{idx_round}.tsv"
            do_retrieve(args)

            args.ranking = os.path.join(
                Run.path,
                args.out_ranking_base,
            )
            args.output = os.path.join(
                Run.path,
                f"train.self.round_{idx_round}.triples",
            )
            get_self_guided_training_w_gold(args)
            rel_link_last_file(
                Run.path,
                "train.self.LAST.triples",
                "train.self.round_*.triples",
            )
            elapsed = float(time.time() - start_time)
            Run.info(f"Time for this round: {elapsed} seconds")
            elapsed = float(time.time() - real_start_time)
            Run.info(f"Actual Time for this round: {elapsed} seconds")
            elapsed = float(time.time() - start_time_0)
            Run.info(f"Total Time elapsed: {elapsed} seconds")


if __name__ == "__main__":
    main()

