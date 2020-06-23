#!/bin/sh

python3 decisiontrees/main.py 70 data/cleaned_processed.cleveland.data -g -n out_cleveland_tree70
python3 neuralnetworks/main.py 70 data/cleaned_processed.cleveland.data -g -n out_cleveland_net70
python3 randomforests/main.py 70 data/cleaned_processed.cleveland.data -g -n out_cleveland_forest70
python3 decisiontrees/main.py 70 data/spambase.data -g -n out_spam_tree70
python3 neuralnetworks/main.py 70 data/spambase.data -g -n out_spam_net70
python3 randomforests/main.py 70 data/spambase.data -g -n out_spam_forest70
