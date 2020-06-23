#!/bin/sh

python3 decisiontrees/main.py 60 data/cleaned_processed.cleveland.data -g -n out_cleveland_tree60
python3 neuralnetworks/main.py 60 data/cleaned_processed.cleveland.data -g -n out_cleveland_net60
python3 randomforests/main.py 60 data/cleaned_processed.cleveland.data -g -n out_cleveland_forest60
python3 decisiontrees/main.py 60 data/spambase.data -g -n out_spam_tree60
python3 neuralnetworks/main.py 60 data/spambase.data -g -n out_spam_net60
python3 randomforests/main.py 60 data/spambase.data -g -n out_spam_forest60
