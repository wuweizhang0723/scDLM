# scDLM - Single-cell DNA Language Model

Install conda environment
```
conda env create --name scDLM --file environment.yml
```

Install the scDLM package
```
conda activate scDLM
pip install -e .
```

## Data
* Google drive: https://drive.google.com/drive/folders/1DvicjZRVsugAhMHMQrHxwMBo_tjPotZ9?usp=sharing
* hg38 reference genome is used.
* Download the data from google drive and create a `data` folder to save it.

## Model Training
* `cd scripts`
* `bash train.sh`