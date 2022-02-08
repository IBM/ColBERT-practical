# import os
# import sys
# import git
# import ujson
import random
from collections import defaultdict

from argparse import ArgumentParser
#from colbert.utils.utils import print_message


MAX_NUM_TRIPLES = 40_000_000


def sample_for_query_1(qid, ranking, positives, depth_negative, max_n_neg, min_n_neg):
    """
        Requires that the ranks are sorted per qid.
    """
    negatives, triples = [], []
    found_positive = False
    n_match = 0

    for pid, rank, *_ in ranking:
        assert rank >= 1, f"ranks should start at 1 \t\t got rank = {rank}"
        # stop if we ever see a positive because we only want to include super hard negatives
        if pid in positives:
            found_positive = True
            if rank == 1:
                n_match += 1
            continue
        if rank > depth_negative:
            # continue instead of break because we want to report n_match
            continue
        if found_positive and len(negatives) >= min_n_neg:
            break
        negatives.append(pid)

    if len(negatives) > max_n_neg:
        negatives = random.sample(negatives, max_n_neg)
    for neg in negatives:
        for positive in positives:
            triples.append((qid, positive, neg))

    return triples, n_match


def sample_for_query_2(
        qid,
        ranking,
        positives,
        depth_negative,
        depth_easy_negative,
        n_neg_hard,
        min_n_neg,
):
    """
        Requires that the ranks are sorted per qid.
    """
    triples = []
    found_positive = False
    n_match = 0

    pids = [r[0] for r in ranking]
    hard_negatives, easy_negatives = [], []
    for pid, rank, *_ in ranking:
        assert rank >= 1, f"ranks should start at 1 \t\t got rank = {rank}"
        if pid in positives:
            found_positive = True
            if rank == 1:
                n_match = 1
            continue
        if rank > depth_negative:
            # continue instead of break because we want to report n_match
            continue
        if not found_positive:
            hard_negatives.append(pid)
        else:
            if rank > depth_easy_negative:
                break
            easy_negatives.append(pid)
    if len(hard_negatives) > n_neg_hard:
        hard_negatives = random.sample(hard_negatives, n_neg_hard)
    n_needed = max(0, min_n_neg - len(hard_negatives))
    if n_needed:
        easy_negatives = random.sample(easy_negatives, n_needed)
    else:
        easy_negatives = []
    for neg in hard_negatives + easy_negatives:
        for positive in positives:
            triples.append((qid, positive, neg))

    return triples, n_match


def load_ranking(file_ranking):
    qid2ranking = defaultdict(list)
    with open(file_ranking) as INF:
        for line in INF:
            a = line.strip().split()
            assert len(a) == 3
            a = [int(x) for x in a]
            qid2ranking[a[0]].append((a[1], a[2]))
    return qid2ranking


def load_positive(file_positive):
    qid2positives = defaultdict(list)
    with open(file_positive) as INF:
        for line in INF:
            a = line.strip().split()
            assert len(a) == 2
            a = [int(x) for x in a]
            qid2positives[a[0]].append(a[1])
    return qid2positives


def get_self_guided_training_w_gold(args):
    qid2rankings = load_ranking(args.ranking)
    qid2positives = load_positive(args.positive)

    triples = []
    non_empty_qids = 0
    n_match = 0

    for processing_idx, qid in enumerate(qid2rankings):
        if args.sample_strategy == 's1':
            new_triples, new_n_match = sample_for_query_1(
                qid,
                qid2rankings[qid],
                qid2positives[qid],
                args.depth_negative,
                args.max_n_neg,
                args.min_n_neg,
            )
        elif args.sample_strategy == 's2':
            new_triples, new_n_match = sample_for_query_2(
                qid,
                qid2rankings[qid],
                qid2positives[qid],
                args.depth_negative,
                args.depth_easy_negative,
                args.n_neg_hard,
                args.min_n_neg,
            )

        non_empty_qids += (len(new_triples) > 0)
        n_match += new_n_match
        triples.extend(new_triples)

        if processing_idx % 10_000 == 0:
            print(f"#> Done with {processing_idx+1} questions!\t\t "
                          f"{str(len(triples) / 1000)}k triples for {non_empty_qids} unqiue QIDs.")

    print(f"#> Sub-sample the triples (if > {MAX_NUM_TRIPLES})..")
    print(f"#> len(triples) = {len(triples)}")

    if len(triples) > MAX_NUM_TRIPLES:
        triples = random.sample(triples, MAX_NUM_TRIPLES)

    print("#> Writing {}M examples to file.".format(len(triples) / 1000.0 / 1000.0))

    with open(args.output, 'w') as f:
        for qid, pos, neg in triples:
            f.write(f'{qid}\t{pos}\t{neg}\n')

    # print('\n\n', args, '\n\n')
    print(args.output)
    ratio_match = n_match / len(qid2rankings)
    print(f'match@1 = {ratio_match}')
    print("#> Done.")


def main():
    random.seed(12345)

    parser = ArgumentParser(description='Create training triples from ranked list.')

    # Input / Output Arguments
    parser.add_argument('--ranking', dest='ranking', required=True, type=str)
    parser.add_argument('--positive', dest='positive', required=True, type=str)
    parser.add_argument('--output', dest='output', required=True, type=str)
    parser.add_argument('--sample_strategy', dest='sample_strategy', required=False, default='s1', type=str)
    parser.add_argument('--depth-', dest='depth_negative', required=True, type=int)
    parser.add_argument('--max_n_neg', dest='max_n_neg', required=False, default=100, type=int)
    parser.add_argument('--min_n_neg', dest='min_n_neg', required=False, default=3, type=int)

    opts = parser.parse_args()
    # assert not os.path.exists(args.output), args.output

    get_self_guided_training_w_gold(opts)


if __name__ == "__main__":
    main()
