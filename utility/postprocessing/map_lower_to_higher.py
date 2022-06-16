import json
import argparse
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_dict_mapping(in_file):
    dict_hi_low = {}
    with open(in_file, 'r', encoding='utf8') as in_f:
        for line in in_f.readlines():
            a = line.strip().split('\t')
            assert len(a) == 2
            dict_hi_low[a[0]] = a[1]
    return dict_hi_low


def lower2higher(result_json, dict_lower_higher_file, out_file, pooling="sum"):
    dict_lower_higher = read_dict_mapping(dict_lower_higher_file)
    all_entries = []
    with open(result_json, 'r') as inf:
        data = json.load(inf)
        for x in data:
            question = x['question']
            preds = x['ctxs']
            higher_preds = defaultdict(float)
            for p in preds:
                lower = p['title']
                if pooling == "sum":
                    higher_preds[dict_lower_higher[lower]] += float(p['score'])
                elif pooling == "max":
                    higher_preds[dict_lower_higher[lower]] = max(higher_preds[dict_lower_higher[lower]], float(p['score']))
                else:
                    raise ValueError("Pooling must be 'sum' or 'max'.")
            ctxs = [{'title': higher, 'score': score} for higher, score in higher_preds.items()]
            x['ctxs'] = ctxs
            all_entries.append(x)
    with open(out_file, 'w') as outf:
        json.dump(all_entries, outf, indent=4)
        logger.info("HIGHERs results written to {}".format(out_file))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-lower_json", type=str, required=True)
    parser.add_argument("-pooling", type=str, required=False, default="sum", choices=["sum", "max"])
    parser.add_argument("-dict_lower_higher", type=str, required=True)
    parser.add_argument("-out_higher", type=str, default=None)

    args = parser.parse_args()
    if not args.out_higher:
        args.out_higher = args.lower_json + '.highers.json'

    lower2higher(args.lower_json, args.dict_lower_higher, args.out_higher, args.pooling)


if __name__ == "__main__":
    main()
