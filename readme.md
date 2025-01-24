# Understanding Model Reprogramming for CLIP via Decoupling Visual Prompts

This is the implementation of our paper submitted to ICML2025.

## Installation
    conda create -n reprogram
    conda activate reprogram
    pip install -r requirement.txt

## Dataset Preparation
### Step 1: Downloading Images
Please follow Appendix A.1 to download the datasets and modify `DOWNSTREAM_PATH = ""` in `cfg.py`.
### Step 2: Downloading Descriptions
Please download the attribute descriptions provided by `AttrVR` and put them under `attributes/`
### Step 3: Generating Causes
Please first enter your API Key in `generate_causes.py`, then run the code:
        
    python generated_attributes.py

## Runing the Code for DVP-cls & DVP-cse

    python experiments/fs_dvp_cls.py --dataset [dataset]
	python experiments/fs_dvp_cse.py --dataset [dataset]

