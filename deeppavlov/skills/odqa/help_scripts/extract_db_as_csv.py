import argparse
import csv

from deeppavlov.dataset_iterators.sqlite_iterator import SQLiteDataIterator

parser = argparse.ArgumentParser()

parser.add_argument("-db_path", help="path to a JSON ranker config", type=str,
                    default='/media/olga/Data/projects/DeepPavlov/download/general_electrics/ge_book.db')
parser.add_argument("-save_path", help="path where the ready csv should be saved", type=str,
                    default='/media/olga/Data/datasets/bhge/ge_book.csv')


def main():
    args = parser.parse_args()
    db_path = args.db_path
    save_path = args.save_path
    iterator = SQLiteDataIterator(db_path)
    with open(save_path, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(['chunk_id', 'chunk'])  # header
        for i in iterator.doc_ids:
            content = iterator.get_doc_content(i)
            writer.writerow([i, content])


if __name__ == "__main__":
    main()
