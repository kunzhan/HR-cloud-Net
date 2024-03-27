#!/bin/bash
# echo "GPU ID is $1, dataset is $2"

python landsat_test.py --gpu $1 --d 38

python landsat_test.py --gpu $1 --d CH

python landsat_test.py --gpu $1 --d spars