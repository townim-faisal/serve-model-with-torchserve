# Serve Toxicity & Sentiment Analysis Model Using `torchserve`

## Commands
command to create toxicity model
```
torch-model-archiver --model-name ToxicClassification --version 1.0 --model-file model.py --serialized-file toxicity_model/model.pt --handler ./model_handler.py --extra-files "./setup_config.json,./index_to_name.json"
```

command to create sentiment analysis model (huggingface)
```
torch-model-archiver --model-name BERTSeqClassification --version 1.0 --serialized-file Transformer_model/pytorch_model.bin --handler ./Transformer_handler_generalized.py --extra-files "Transformer_model/config.json,./setup_config.json,./index_to_name.json"
```

commands to serve all model
```
torchserve --start --ncs --model-store ./models --models toxicity=ToxicClassification.mar sentiment=BERTSeqClassification.mar --ts-config ./config.properties
```

serve only toxicity model
```
torchserve --start --ncs --model-store model_store --models toxicity=ToxicClassification.mar --ts-config ./config.properties
```

serve only sentiment analysis model (huggingface)
```
torchserve --start --model-store model_store --models sentiment=BERTSeqClassification.mar --ncs --ts-config ./config.properties
```