import argparse
from os import path

from deeppavlov.core.common.file import read_json

parser = argparse.ArgumentParser()

parser.add_argument("-lists_path", help="path to a JSON file with parsed tables", type=str,
                    default='/media/olga/Data/datasets/bhge/lists/lists_with_questions.json')
parser.add_argument("-save_path", help="path to a JSON file with parsed tables", type=str,
                    default='/media/olga/Data/datasets/bhge/lists/lists_as_txt')


def main():
    args = parser.parse_args()
    lists_path = args.lists_path
    save_path = args.save_path
    lists = read_json(lists_path)
    for i, l in enumerate(lists):
        list2text = '\n'.join(l['list']['headerPath'])
        # list2text = l['list']['headerPath']
        # list2text += l['list']['items']
        # list2text = '\n'.join(list2text)

        with open(path.join(save_path, f'{i}.txt'), 'w') as fout:
            fout.write(list2text)

    print('Done!')


if __name__ == '__main__':
    main()
