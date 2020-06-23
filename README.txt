REQUIRED DEPENDENCIES:
graphviz
six
scikit-learn
matplotlib
pydotplus
numpy

COMMANDS RUN:
sudo apt install graphviz
pip install six
pip install scikit-learn
python3 -m pip install -U matplotlib
pip install pydotplus

-g turns no graphs
-n specifies an output file

REQUIRED INPUT:
python3 [folder]/main.py [test_split percentage] data/[data_file] -g -n [output_file_name]

EXAMPLE INPUT:
python3 decisiontrees/main.py 30 data/cleaned_processed.cleveland.data -g -n out_cleveland_tree30 
python3 neuralnetworks/main.py 30 data/cleaned_processed.cleveland.data -g -n out_cleveland_net30
python3 randomforests/main.py 30 data/cleaned_processed.cleveland.data -g -n out_cleveland_forest30

python3 decisiontrees/main.py 30 data/spambase.data -g -n out_spam_tree30 
python3 neuralnetworks/main.py 30 data/spambase.data -g -n out_spam_net30
python3 randomforests/main.py 30 data/spambase.data -g -n out_spam_forest30

SCRIPTS:
./run_all_jobs

This script will run all of the scripts in shell_scripts.
WARNING: This takes a long time.

Alternatively, you can run groups of scripts manually as they're separated by
their test split percentages in the shell_scripts folder.
