
Docker image uri:  
cpu
763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:2.0.0-transformers4.28.1-cpu-py310-ubuntu20.04
```763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:2.0.0-transformers4.28.1-cpu-py310-ubuntu20.04```
GPU
```763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04```


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
export IMAGE_URI=763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04
export MODEL_ARTIFACTS=s3://llama-weights-us/labse-model/model.tar.gz

python create_model_and_inference.py --model_name labse-model \
--image_uri $IMAGE_URI \
--model_artifact_path $MODEL_ARTIFACTS \
--batch_size 64 \
--strategy MultiRecord \
--create_model \
--instance_type  ml.g4dn.xlarge  \
--input_data s3://llama-weights-us/input_data/ \
--output_data s3://llama-weights-us/output_data/ \
--serve 
```
This command will create model name with args combination and transform job.  
`labse-model-64-MultiRecord-csv`

### For creating batch transform job manully use this command:
```
python create_model_and_inference.py --model_name labse-model-64-MultiRecord-csv \
--input_data s3://llama-weights-us/input_data/  \
--output_data s3://llama-weights-us/output_data/ \
--serve
```


