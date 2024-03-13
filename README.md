# MultimodalExpression

## Installation

This project requires Python 3.9 or higher. It is recommended to create a virtual environment using conda and then install the dependencies using pip.

1. Create a virtual environment and activate it:

```bash
conda create -n multimodal_expr python=3.9
```

```bash
conda activate multimodal_expr
```


2. Clone this repository:
```bash
git clone https://github.com/doveppp/MultimodalExpression.git
```

3. Install dependencies:
```bash
cd MultimodalExpression
pip install -r requirements.txt
```

## Download Dataset and trained models

```bash
gdown 1EccZDCyPopiKfxBRxK-dDz8h1ed6noQZ
tar -xzvf MultimodalExpressionData.tar.gz
```

## Usage
### MultimodalExpression model

use 
`python train.py BertDeepLncLocMultimodalExpression --seq` to train BertDeepLncLocMultimodalExpression with promoter data.
use `python train.py Conv2dDeepLncLocMultimodalExpression --seq` to train Conv2dDeepLncLocMultimodalExpression with promoter data.

use `python train.py BertDeepLncLocMultimodalExpression --seq --hl` to train BertDeepLncLocMultimodalExpression with promoter data and halflife data.

use `python train.py BertDeepLncLocMultimodalExpression --seq --hl --tf` to train BertDeepLncLocMultimodalExpression with promoter data , halflife data and transcription factor data.

### predict tf with promoter data and halflife data model

```
python train.py TransformerSeqHalflifeToTF
```
note: The parameters --seq, --hl, and --tf do not work in this model.

### train bert
You can use the pre-trained BERT model we provide, or you can train your own.

```bash
python run_mlm.py config_bert_train/bert_k3_to_k7.json
```

The trained BERT model will be placed in the directory `trained_models/bert/promoter_k3_to_k7`.

note: The BERT model in the `models/bert/promoter_k3_to_k7/` directory is initialized completely randomly.

After training is completed, you can change the BERT_PATH in the `models/BertDeepLncLocMultimodalExpression.py` file to the path of your trained BERT model.