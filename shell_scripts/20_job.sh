#!/bin/sh

python3 decisiontrees/main.py 20 data/cleaned_processed.cleveland.data -g -n out_cleveland_tree20
python3 neuralnetworks/main.py 20 data/cleaned_processed.cleveland.data -g -n out_cleveland_net20
python3 randomforests/main.py 20 data/cleaned_processed.cleveland.data -g -n out_cleveland_forest20
python3 decisiontrees/main.py 20 data/spambase.data -g -n out_spam_tree20
python3 neuralnetworks/main.py 20 data/spambase.data -g -n out_spam_net20
python3 randomforests/main.py 20 data/spambase.data -g -n out_spam_forest20

