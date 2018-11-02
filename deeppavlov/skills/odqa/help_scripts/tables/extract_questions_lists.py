import argparse
from os import path
import csv

from deeppavlov.core.common.file import read_json

parser = argparse.ArgumentParser()

parser.add_argument("-lists_path", help="path to a JSON file with parsed tables", type=str,
                    default='/media/olga/Data/datasets/bhge/lists/lists_with_questions.json')
parser.add_argument("-save_path", help="path to a JSON file with parsed tables", type=str,
                    default='/media/olga/Data/datasets/bhge/lists/questions_to_numbers.csv')


def main():
    args = parser.parse_args()
    lists_path = args.lists_path
    save_path = args.save_path
    lists = read_json(lists_path)
    with open(save_path, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(['question', 'number'])  # header
        for i, l in enumerate(lists):
            writer.writerow([l['question'], i])

    print('Done!')


if __name__ == '__main__':
    main()
