import argparse
import sqlite3
import re
import unicodedata
import json


parser = argparse.ArgumentParser()

parser.add_argument("-input_path",
                    type=str,
                    default="/home/olga/Documents/DrillingFluidsReferenceManual_cleaned")
parser.add_argument("-output_path",
                    type=str,
                    default="/home/olga/Documents/DrillingFluidsReferenceManual_cleaned_prepared_4ranker.json")


def main():
    args = parser.parse_args()
    with open(args.input_path) as fin:
        contents = fin.read()

    paragraphs = [p for p in contents.split('\n') if len(p.split()) > 12]
    par_list = [{'title': i, 'text': p} for i, p in enumerate(paragraphs)]

    with open(args.output_path, 'w') as fout:
        json.dump(par_list, fout)

    print('Done!')


if __name__ == "__main__":
    main()
