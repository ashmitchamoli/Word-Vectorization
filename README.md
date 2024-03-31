# Execution Instructions
Simply execute the python files from the root directory.

# Loading Pre-trained Embeddings
Place the saved word embedding (.pkl) files into `/model_cache/(svd/word2vec)/` folder. The embeddings will automatically be loaded when the `train()` method of the model is called. If you wish to retrain the model, set `retrain = True` in the `train()` method of the model. 

# Loading Pre-trained Classifier
Place the saved classifier (.pkl) file into `/model_cache/lstm_classifier/` folder. The classifier will automatically be loaded when the `train()` method of the model is called. If you wish to retrain the model, set `retrain = True` in the `train()` method of the model.