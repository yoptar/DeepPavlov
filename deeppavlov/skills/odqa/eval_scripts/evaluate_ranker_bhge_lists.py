"""
Count a ranker recall.
Need:
1. A ranker config (with rank, text, score, text_id API) for a specific domain (eg. "en_drones")
2. QA dataset for this domain.
"""

import argparse
import time
import unicodedata
import logging
import csv
import re
import string
import matplotlib.pylab as plt

from deeppavlov.core.common.file import read_json
from deeppavlov.core.commands.infer import build_model_from_config

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
file = logging.FileHandler('../eval_logs/ranker_ensemble_bhge_tables.log')
file.setFormatter(fmt)
logger.addHandler(file)

parser = argparse.ArgumentParser()

parser.add_argument("-config_path", help="path to a JSON ranker config", type=str,
                    default='../../../../deeppavlov/configs/odqa/bhge/bhge_retrieval_demo_lists_ensemble.json')
parser.add_argument("-dataset_path", help="path to a QA TSV dataset", type=str,
                    default='/media/olga/Data/datasets/bhge/lists/questions_to_numbers.csv')


def normalize(s: str):
    return unicodedata.normalize('NFD', s)


def instance_score_by_text(answers, texts):
    formatted_answers = [normalize(a.strip().lower()) for a in answers]
    formatted_texts = [normalize(text.lower()) for text in texts]
    for a in formatted_answers:
        for doc_text in formatted_texts:
            if doc_text.find(a) != -1:
                return 1
    return 0

# def instance_score_by_table_id(answers, texts):
#     formatted_answers = [normalize(a.strip().lower()) for a in answers]
#     formatted_texts = [normalize(text.lower()) for text in texts]
#     for a in formatted_answers:
#         for doc_text in formatted_texts:
#             if doc_text.find(a) != -1:
#                 return 1
#     return 0


def read_csv(csv_path):
    output = []
    with open(csv_path) as fin:
        reader = csv.reader(fin, delimiter=';')
        next(reader)  # skip header
        for item in reader:
            output.append({'question': item[0],
                           'number': item[1]})
    return output


def main():
    args = parser.parse_args()
    config = read_json(args.config_path)
    ranker = build_model_from_config(config)  # chainer
    dataset = read_csv(args.dataset_path)
    # dataset = dataset[:10]

    qa_dataset_size = len(dataset)
    logger.info('QA dataset size: {}'.format(qa_dataset_size))
    # n_queries = 0  # DEBUG
    start_time = time.time()
    TEXT_IDX = 1

    try:
        mapping = {}

        ranker_answers = ranker([i['question'] for i in dataset])
        returned_db_size = len(ranker_answers[0])
        logger.info("Returned DB size: {}".format(returned_db_size))

        for n in range(1, returned_db_size + 1):
            correct_answers = 0
            for qa, ranker_answer in zip(dataset, ranker_answers):
                true_id = int(qa['number'])
                pred_ids = ranker_answer[:n]
                correct_answers += int(true_id in pred_ids)
                # correct_answer = qa['answer']
                # texts = [answer[TEXT_IDX] for answer in ranker_answer[:n]]
                # correct_answers += instance_score([correct_answer], texts)
            print(correct_answers)
            total_score_on_top_i = correct_answers / qa_dataset_size
            logger.info(
                'Recall for top {}: {}'.format(n, total_score_on_top_i))
            mapping[n] = total_score_on_top_i

        logger.info("Completed successfully in {} seconds.".format(time.time() - start_time))
        logger.info("Quality mapping: {}".format(mapping))

    except Exception as e:
        logger.exception(e)
        logger.info("Completed with exception in {} seconds".format(time.time() - start_time))
        raise


if __name__ == "__main__":
    main()

