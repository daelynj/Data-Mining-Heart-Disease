#!/bin/sh

python3 decisiontrees/main.py 10 data/cleaned_processed.cleveland.data -g -n out_cleveland_tree10 
python3 neuralnetworks/main.py 10 data/cleaned_processed.cleveland.data -g -n out_cleveland_net10
python3 randomforests/main.py 10 data/cleaned_processed.cleveland.data -g -n out_cleveland_forest10
python3 decisiontrees/main.py 10 data/spambase.data -g -n out_spam_tree10 
python3 neuralnetworks/main.py 10 data/spambase.data -g -n out_spam_net10
python3 randomforests/main.py 10 data/spambase.data -g -n out_spam_forest10
