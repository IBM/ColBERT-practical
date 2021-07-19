import os
import json

from contextlib import contextmanager
from colbert.utils.utils import print_message, NullContextManager
from colbert.utils.runs import Run


class RankingLogger():
    def __init__(self, directory, qrels=None, log_scores=False):
        self.directory = directory
        self.qrels = qrels
        self.filename, self.also_save_annotations = None, None
        self.log_scores = log_scores

    @contextmanager
    def context(self, filename, also_save_annotations=False, also_save_json=False):
        assert self.filename is None
        assert self.also_save_annotations is None

        filename = os.path.join(self.directory, filename)
        self.filename, self.also_save_annotations = filename, also_save_annotations

        print_message("#> Logging ranked lists to {}".format(self.filename))

        with open(filename, 'w') as f:
            self.f = f
            with (open(filename + '.json', 'w') if also_save_json else NullContextManager()) as j:
                self.j = j
                self.j_buffer = []
                with (open(filename + '.annotated', 'w') if also_save_annotations else NullContextManager()) as g:
                    self.g = g
                    try:
                        yield self
                    finally:
                        pass

    def log_json(self):
        assert self.j and self.j_buffer
        self.j.write(json.dumps(self.j_buffer, indent=4) + "\n")

    def log(self, qid, ranking, is_ranked=True, print_positions=[], queries=None, titles=None):
        print_positions = set(print_positions)

        f_buffer = []
        g_buffer = []
        ctxs = []

        for rank, (score, pid, passage) in enumerate(ranking):
            is_relevant = self.qrels and int(pid in self.qrels[qid])
            rank = rank+1 if is_ranked else -1

            possibly_score = [score] if self.log_scores else []

            f_buffer.append('\t'.join([str(x) for x in [qid, pid, rank] + possibly_score]) + "\n")
            if self.g:
                g_buffer.append('\t'.join([str(x) for x in [qid, pid, rank, is_relevant]]) + "\n")
            if self.j:
                ctxs.append(
                    {
                        "id": pid,
                        "title": titles[pid] if titles else "",
                        "score": score,
                    }
                )

            if rank in print_positions:
                prefix = "** " if is_relevant else ""
                prefix += str(rank)
                print("#> ( QID {} ) ".format(qid) + prefix + ") ", pid, ":", score, '    ', passage)

        self.j_buffer.append(
            {
                "question": queries[qid] if queries else qid,
                "ctxs": ctxs,
            }
        )
        self.f.write(''.join(f_buffer))
        if self.g:
            self.g.write(''.join(g_buffer))
