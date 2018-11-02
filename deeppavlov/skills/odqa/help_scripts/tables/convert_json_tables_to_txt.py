import argparse
from os import path

from deeppavlov.core.common.file import read_json

parser = argparse.ArgumentParser()

parser.add_argument("-tables_path", help="path to a JSON file with parsed tables", type=str,
                    default='/media/olga/Data/datasets/bhge/tables/corrected_tables_v2.json')
parser.add_argument("-save_path", help="path to a JSON file with parsed tables", type=str,
                    default='/media/olga/Data/datasets/bhge/tables/tables_as_txt_v2')


def main():
    args = parser.parse_args()
    tables_path = args.tables_path
    save_path = args.save_path
    tables = read_json(tables_path)
    for i, t in enumerate(tables):
        table2text = []
        for r in t['rows']:
            table2text += r
        for hp in t['headerPath']:
            table2text.append(hp)
        for c in t['columns']:
            table2text.append(c)
        table2text.append(' '.join(t['title'].split('.')[1:]))

        with open(path.join(save_path, f'{i}.txt'), 'w') as fout:
            fout.write('\n'.join(table2text))

    print('Done!')


if __name__ == '__main__':
    main()
