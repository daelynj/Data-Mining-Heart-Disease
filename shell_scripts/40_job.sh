#!/bin/sh

python3 decisiontrees/main.py 40 data/cleaned_processed.cleveland.data -g -n out_cleveland_tree40 
python3 neuralnetworks/main.py 40 data/cleaned_processed.cleveland.data -g -n out_cleveland_net40
python3 randomforests/main.py 40 data/cleaned_processed.cleveland.data -g -n out_cleveland_forest40
python3 decisiontrees/main.py 40 data/spambase.data -g -n out_spam_tree40 
python3 neuralnetworks/main.py 40 data/spambase.data -g -n out_spam_net40
python3 randomforests/main.py 40 data/spambase.data -g -n out_spam_forest40
