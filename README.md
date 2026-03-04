# EndoScribe


## Installation

```sh
pip install torch==2.5.1 torchvision==0.20.1 transformers==4.57.3 qwen_vl_utils accelerate translate
```

## Download EndoScribe Trained Models

Please download the trained EndoScribe model and place it in the `./output` directory.

```sh
mkdir -p output
cd output
# Download EndoScribe trained models
git lfs clone https://huggingface.co/xieqiang/EndoScribe
cd ..
```

## Download Sample Data

Please download the sample data and place it in the `./sample_data` directory.

```sh
mkdir -p sample_data
cd sample_data
# Download Sample Data
git lfs clone https://huggingface.co/datasets/xieqiang/EndoScribe_sample_data
cd ..
```

## Inference

```sh
python inference.py
```
