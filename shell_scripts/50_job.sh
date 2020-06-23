#!/bin/sh

python3 decisiontrees/main.py 50 data/cleaned_processed.cleveland.data -g -n out_cleveland_tree50
python3 neuralnetworks/main.py 50 data/cleaned_processed.cleveland.data -g -n out_cleveland_net50
python3 randomforests/main.py 50 data/cleaned_processed.cleveland.data -g -n out_cleveland_forest50
python3 decisiontrees/main.py 50 data/spambase.data -g -n out_spam_tree50
python3 neuralnetworks/main.py 50 data/spambase.data -g -n out_spam_net50
python3 randomforests/main.py 50 data/spambase.data -g -n out_spam_forest50
