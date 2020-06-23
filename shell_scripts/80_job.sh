#!/bin/sh

python3 decisiontrees/main.py 80 data/cleaned_processed.cleveland.data -g -n out_cleveland_tree80
python3 neuralnetworks/main.py 80 data/cleaned_processed.cleveland.data -g -n out_cleveland_net80
python3 randomforests/main.py 80 data/cleaned_processed.cleveland.data -g -n out_cleveland_forest80
python3 decisiontrees/main.py 80 data/spambase.data -g -n out_spam_tree80
python3 neuralnetworks/main.py 80 data/spambase.data -g -n out_spam_net80
python3 randomforests/main.py 80 data/spambase.data -g -n out_spam_forest80
