
__version__ = '0.1'
__author__ = 'Hui Wan'

import json
import argparse
import logging
from collections import defaultdict

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


def read_query_doc_tsv(file_name):
    questions = []
    doc_ids = defaultdict(list)
    with open(file_name, 'r') as f_in:
        for line in f_in.readlines():
            a = line.strip().split('\t')
            assert len(a) > 0
            questions.append(a[0])
            if len(a) == 2:
                doc_ids[a[0]].append(a[1])
            elif len(a) >= 3:
                doc_ids[a[0]].insert(int(a[2]) - 1, a[1])
                # doc_ids[a[0]].append(a[1])
            elif len(a) == 1:
                doc_ids[a[0]].append("")
    doc_ids = [doc_ids[q] for q in questions]
    return questions, doc_ids


def eval(ref_file, hyp_file, out_json_file, filter=None, reverse_filter=False, thres=THRES, sort=False):
    if hyp_file.endswith(".json"):
        question_hyp, doc_ids_hyp = read_query_docs_json(hyp_file, sort=sort)
    else:
        question_hyp, doc_ids_hyp = read_query_doc_tsv(hyp_file)

    if ref_file.endswith(".json") or ref_file.endswith(".json.all") :
        question_ref, doc_ids_ref = read_query_docs_json(ref_file, sort=sort)
    else:
        question_ref, doc_ids_ref = read_query_doc_tsv(ref_file)

    print(f"{len(question_ref)} {len(question_hyp)} {len(doc_ids_ref)} {len(doc_ids_hyp)}")
    assert len(question_ref) == len(question_hyp) == len(doc_ids_ref) == len(doc_ids_hyp)

    filter_qids = []
    if filter:
        with open(filter) as FILTER_IN:
            data = json.load(FILTER_IN)
            filter_qids = [d['qid'] for d in data]

    recalls = {}
    mrrs = {}
    for k in thres:
        out_json = []
        n_hit = 0
        rr = 0
        for i in range(len(question_ref)):
            if filter_qids and not reverse_filter and i not in filter_qids:
                continue
            if filter_qids and reverse_filter and i in filter_qids:
                continue
            #assert question_ref[i].strip() == question_hyp[i].strip(), question_ref[i].strip() + '|' + question_hyp[i].strip()
            hit_rank = k+1
            for rank, h in enumerate(doc_ids_hyp[i][:k]):
                if h in doc_ids_ref[i]:
                    n_hit += 1
                    hit_rank = rank + 1
                    rr += 1/(rank+1)
                    break
            out_json.append(
                {
                    'qid': i,
                    # i in [0, k-1]: match at i + 1, i=k: miss
                    'rank_of_gold': hit_rank,
                    'question': question_ref[i].strip(),
                    'prediction': doc_ids_hyp[i][0] if doc_ids_hyp[i] else [],
                    'gold': doc_ids_ref[i],
                }
            )
        recall = n_hit / len(out_json)
        mrr = rr / len(out_json)
        recalls[k] = recall
        mrrs[k] = mrr
        print("Match@{} is {}  {}/{}".format(k, recall, n_hit, len(out_json)))
        print("MRR@{} is {}".format(k, mrr))
        with open(out_json_file + '.{}'.format(k), 'w') as JOUT:
            json.dump(out_json, JOUT, indent=4)
    return recalls, mrrs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-hyp", type=str)
    parser.add_argument("-ref", type=str)
    parser.add_argument("-out_json", type=str)
    parser.add_argument("-filter", required=False, type=str)
    parser.add_argument("-reverse_filter", action='store_true')

    args = parser.parse_args()
    result_file = args.hyp
    ref_file = args.ref
    if not args.out_json:
        args.out_json = args.hyp + '.eval.json'

    eval(ref_file=ref_file,
         hyp_file=result_file,
         out_json_file=args.out_json,
         filter=args.filter if args.filter else None,
         reverse_filter=args.reverse_filter,
         sort=True)


if __name__ == "__main__":
    main()
