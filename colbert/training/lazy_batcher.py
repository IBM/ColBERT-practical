# import os
# import ujson
import random

from functools import partial
from colbert.utils.utils import print_message
from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer, tensorize_triples

from colbert.utils.runs import Run


class LazyBatcher():
    def __init__(self, args, rank=0, nranks=1):
        self.rank, self.nranks = rank, nranks
        self.bsize, self.accumsteps = args.bsize, args.accumsteps

        self.query_tokenizer = QueryTokenizer(args.query_maxlen, args.model_name)
        self.doc_tokenizer = DocTokenizer(args.doc_maxlen, args.model_name)
        self.tensorize_triples = partial(tensorize_triples, self.query_tokenizer, self.doc_tokenizer)
        self.position = 0

        self.triples = self._load_triples(args.triples, rank, nranks)
        # self.shuffle()
        self.reader = open(args.triples, mode='r', encoding="utf-8")
        self.length = len(self.reader.readlines())
        self.queries = self._load_queries(args.queries)
        self.collection = self._load_collection(args.collection)

    def shuffle(self):
        print_message("#> Shuffling triples...")
        random.shuffle(self.triples)

    def _load_triples(self, path, rank, nranks):
        """
        NOTE: For distributed sampling, this isn't equivalent to perfectly uniform sampling.
        In particular, each subset is perfectly represented in every batch! However, since we never
        repeat passes over the data, we never repeat any particular triple, and the split across
        nodes is random (since the underlying file is pre-shuffled), there's no concern here.
        """
        print_message("#> Loading triples...")

        triples = []

        with open(path) as f:
            for line_idx, line in enumerate(f):
                if line_idx % nranks == rank:
                    # qid, pos, neg = ujson.loads(line)
                    qid, pos, neg = line.strip().split('\t')
                    triples.append((int(qid), int(pos), int(neg)))

        return triples

    def _load_queries(self, path):
        print_message("#> Loading queries...")

        queries = {}

        with open(path) as f:
            for line in f:
                qid, query = line.strip().split('\t')
                qid = int(qid)
                queries[qid] = query

        return queries

    def _load_collection(self, path):
        print_message("#> Loading collection...")

        collection = []

        with open(path) as f:
            for line_idx, line in enumerate(f):
                pid, passage, *other = line.strip().split('\t')
                assert pid == 'id' or int(pid) == line_idx

                if len(other) >= 1:
                    title, *_ = other
                    passage = title + ' | ' + passage
                collection.append(passage)

        return collection

    def __iter__(self):
        return self

    def __len__(self):
        return self.length

    def __next__(self):
        queries, positives, negatives = [], [], []

        for line_idx in range(self.bsize * self.nranks):
            if (self.position + line_idx) % self.nranks != self.rank:
                continue

            real_line_idx = (self.position + line_idx) % len(self.triples)
            query, pos, neg = self.triples[real_line_idx]
            query, pos, neg = self.queries[query], self.collection[pos], self.collection[neg]

            queries.append(query)
            positives.append(pos)
            negatives.append(neg)
            # self.position += 1

        self.position += line_idx + 1

        Run.info("Queries: {}  rank: {}".format(queries, self.rank))

        return self.collate(queries, positives, negatives)

    def collate(self, queries, positives, negatives):
        assert len(queries) == len(positives) == len(negatives) == self.bsize

        return self.tensorize_triples(queries, positives, negatives, self.bsize // self.accumsteps)

    def skip_to_batch(self, batch_idx, intended_batch_size):
        Run.warn(f'Skipping to batch #{batch_idx} (with intended_batch_size = {intended_batch_size}) for training.')
        self.position = intended_batch_size * batch_idx
