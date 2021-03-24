# Code for running pretraining and finetuning of Chinese BERT model

Note that we split a fraction of the original CLUE training set and use as the dev set, we choose checkpoints based on results of that dev set and evaluate on the original CLUE dev set as the test set.

You can use ```split_data.py``` to do the dev set splitting, but remember to keep the random seed so that we can all reproduct the same splitting and results.
