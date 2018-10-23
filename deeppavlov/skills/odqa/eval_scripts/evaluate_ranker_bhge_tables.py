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
                    default='../../../../deeppavlov/configs/odqa/bhge/bhge_retrieval_demo_tables_ensemble.json')
parser.add_argument("-dataset_path", help="path to a QA TSV dataset", type=str,
                    default='/media/olga/Data/datasets/bhge/tables/questions_v2_answers.tsv')
parser.add_argument("-output_path", help="path to a QA TSV dataset", type=str,
                    default='/media/olga/Data/datasets/bhge/tables/tables_predictions.csv')


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


def read_tsv(tsv_path):
    output = []
    with open(tsv_path) as fin:
        reader = csv.reader(fin)
        for item in reader:
            try:
                output.append({'table_id': item[0],
                               'question': item[1],
                               'answer': item[2]})
            except Exception:
                logger.info("Exception in read_csv()")
    return output


def main():
    args = parser.parse_args()
    config = read_json(args.config_path)
    ranker = build_model_from_config(config)  # chainer
    dataset = read_tsv(args.dataset_path)
    output_path = args.output_path
    # dataset = dataset[:10]

    qa_dataset_size = len(dataset)
    logger.info('QA dataset size: {}'.format(qa_dataset_size))
    # n_queries = 0  # DEBUG
    start_time = time.time()
    TEXT_IDX = 1

    try:
        mapping = {}

        questions = [i['question'] for i in dataset]
        ranker_answers = ranker(questions)
        returned_db_size = len(ranker_answers[0])
        logger.info("Returned DB size: {}".format(returned_db_size))

        with open(output_path, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            for n in range(1, returned_db_size + 1):
                correct_answers = 0
                i = 0
                for qa, ranker_answer in zip(dataset, ranker_answers):
                    true_id = int(qa['table_id'])
                    pred_ids = ranker_answer[:n]
                    correct_answers += int(true_id in pred_ids)
                    # correct_answer = qa['answer']
                    # texts = [answer[TEXT_IDX] for answer in ranker_answer[:n]]
                    # correct_answers += instance_score([correct_answer], texts)
                    if n == 1:
                        writer.writerow([true_id, questions[i], *pred_ids])
                    i += 1
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

