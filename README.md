### Serving or create model manually
you can create model and start batch transform job in one command.  
### Run Locally
install dependencies 
run
```python debug/batch_manager.py```


### For model creation and start `TransformJob` on GPU
- default model is sentence-transformers/LaBSE
to change the model send the sentence-transformers model full name - 'sentence-transformers/all-MiniLM-L6-v2'
use this commands:
```
export MODEL_ARTIFACTS=s3://llama-weights-us/labse-model/model.tar.gz

python create_model_and_inference.py \
--sg_model_name all-mini \ 
--model_artifact_path $MODEL_ARTIFACTS \
--batch_size 64 \
--strategy MultiRecord \
--create_model \
--instance_type  ml.g4dn.xlarge  \
--input_data s3://llama-weights-us/input_data/ \
--output_data s3://llama-weights-us/output_data/ \
--st_model all-MiniLM-L12-v2 \
--serve 
```


### For creating batch transform job manully use this command:
```
python create_model_and_inference.py \
--model_name all-mini \
--input_data s3://llama-weights-us/input_data/  \
--output_data s3://llama-weights-us/output_data/ \
--serve
```


