#!/bin/bash

set -e

\rm -rf data tmp out log
mkdir data tmp out log

parallel -a resources/threhsolds.obs.tsv --colsep '\t' bash run_edited.sh {1} {2} {3}
