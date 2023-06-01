# Fine-tuning Open-source LLaMa with QLoRa Scripts Repository
This repository contains all scripts and resources used in the article 'How to fine-tune an open-source LLaMa using QLoRa'. 

## Install pytorch
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```
## Install GPTQ-for-LLaMa
```
mkdir repositories
cd repositories
git clone git@github.com:mzbac/GPTQ-for-LLaMa.git

cd GPTQ-for-LLaMa
pip install -r requirements.txt
```

## Installation QLoRa dependencies
In the root directory
```
pip install -U -r requirements.txt
```

## Fine-tune the model
Update the dataset configuration based on your data format here -> https://github.com/mzbac/qlora-fine-tune/blob/main/qlora.py#L521-L527
```
python qlora.py --model_name_or_path TheBloke/wizardLM-13B-1.0-fp16 --dataset my-data --bf16
```

## Inference with LoRa adapters

```
python inference.py
```
Note: Change the model_name and adapters_name accordingly

## Merge LoRa adapters back to base model

```
python merge_peft_adapters.py --device cpu --base_model_name_or_path TheBloke/wizardLM-13B-1.0-fp16 --peft_model_path ./output/checkpoint-2250/adapter_model --output_dir ./merged_models/
```

## Quantization Model
```
python repositories/GPTQ-for-LLaMa/llama.py ${MODEL_DIR} c4 --wbits 4 --true-sequential --groupsize 128 --save_safetensors {your-model-name}-no-act-order-4bit-128g.safetensors
```

