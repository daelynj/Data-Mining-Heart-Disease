#!/bin/sh

python3 decisiontrees/main.py 90 data/cleaned_processed.cleveland.data -g -n out_cleveland_tree90
python3 neuralnetworks/main.py 90 data/cleaned_processed.cleveland.data -g -n out_cleveland_net90
python3 randomforests/main.py 90 data/cleaned_processed.cleveland.data -g -n out_cleveland_forest90
python3 decisiontrees/main.py 90 data/spambase.data -g -n out_spam_tree90
python3 neuralnetworks/main.py 90 data/spambase.data -g -n out_spam_net90
python3 randomforests/main.py 90 data/spambase.data -g -n out_spam_forest90
