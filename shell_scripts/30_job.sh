#!/bin/sh

python3 decisiontrees/main.py 30 data/cleaned_processed.cleveland.data -g -n out_cleveland_tree30 
python3 neuralnetworks/main.py 30 data/cleaned_processed.cleveland.data -g -n out_cleveland_net30
python3 randomforests/main.py 30 data/cleaned_processed.cleveland.data -g -n out_cleveland_forest30
python3 decisiontrees/main.py 30 data/spambase.data -g -n out_spam_tree30 
python3 neuralnetworks/main.py 30 data/spambase.data -g -n out_spam_net30
python3 randomforests/main.py 30 data/spambase.data -g -n out_spam_forest30

