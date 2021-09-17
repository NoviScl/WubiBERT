# Code for running pretraining and finetuning of Chinese BERT model

Model checkpoints available at: https://huggingface.co/CLS/WubiBERT_models/tree/main That repo only contains the model checkpoints, the config and tokenizer files are in this repo, which you should load locally. 

More details about how to finetune various versions of models will be added here soon.



Note that we split a fraction of the original CLUE training set and use as the dev set, we choose checkpoints based on results of that dev set and evaluate on the original CLUE dev set as the test set.

You can use ```split_data.py``` to do the dev set splitting, but remember to keep the random seed so that we can all reproduce the same splitting and results.
