import sys
import pandas as pd
import argparse


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-t", "--tf", dest="tf", required=True, help="tf name")
    args = parser.parse_args()

    file = pd.read_csv("out/pbs_"+args.tf+".pred.tsv",sep='\t',header=None)
    file[file[2]==args.tf].to_csv("out/pbs_"+args.tf+".pred.tsv",sep='\t',header=None,index=False,line_terminator='\n')

if __name__ == "__main__":
    sys.exit(main())