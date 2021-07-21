
__version__ = '0.1'
__author__ = 'Hui Wan'

import json
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

THRES = [1, 3, 5, 10]


def read_query_docs_json(file_name, sort=False):
    questions = []
    doc_ids = []
    with open(file_name, 'r') as f_in:
        data = json.load(f_in)
        for d in data:
            questions.append(d['question'])
            ctxs = d['ctxs']
            if sort:
                ctxs = sorted(ctxs, key=lambda x: x["score"], reverse=True)
            doc_ids.append([ctx["title"] for ctx in ctxs])
    return questions, doc_ids


def read_query_doc_ref_tsv(file_name):
    questions = []
    doc_ids = []
    with open(file_name, 'r') as f_in:
        for line in f_in.readlines():
            a = line.strip().split('\t')
            assert len(a) == 2
            questions.append(a[0])
            doc_ids.append([a[1]])
    return questions, doc_ids


def eval(ref_file, hyp_file, ref_format='json', thres=THRES, sort=False):
    if ref_format == "json":
        question_ref, doc_ids_ref = read_query_docs_json(ref_file, sort=sort)
    else:
        question_ref, doc_ids_ref = read_query_doc_ref_tsv(ref_file)
    question_hyp, doc_ids_hyp = read_query_docs_json(hyp_file, sort=sort)
    assert len(question_ref) == len(question_hyp) == len(doc_ids_ref) == len(doc_ids_hyp)
    recalls = {}
    mrrs = {}
    for k in thres:
        n_hit = 0
        rr = 0
        for i in range(len(question_ref)):
            assert question_ref[i].strip() == question_hyp[i].strip()
            for rank, h in enumerate(doc_ids_hyp[i][:k]):
                if h in doc_ids_ref[i]:
                    n_hit += 1
                    rr += 1/(rank+1)
                    break
        recall = n_hit / len(doc_ids_ref)
        mrr = rr / len(doc_ids_ref)
        recalls[k] = recall
        mrrs[k] = mrr
        print("Match@{} is {}".format(k, recall))
        print("MRR@{} is {}".format(k, mrr))
    return recalls, mrrs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-hyp", required=True, type=str)
    parser.add_argument("-ref", required=False, type=str)
    parser.add_argument("-ref_tsv", required=False, type=str)

    args = parser.parse_args()
    if args.ref:
        ref_file = args.ref
        ref_format = 'json'
    elif args.ref_tsv:
        ref_file = args.ref_tsv
        ref_format = 'tsv'
    else:
        assert args.ref or args.ref_tsv
    result_file = args.hyp

    eval(ref_file=ref_file, hyp_file=result_file, ref_format=ref_format, sort=True)


if __name__ == "__main__":
    main()
