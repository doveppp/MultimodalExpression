import argparse
import os


parser = argparse.ArgumentParser()


parser.add_argument("model", type=str, default="BertDeepLncLocMultimodalExpression", help="Model name")
parser.add_argument("--track", default=False, action="store_true")
parser.add_argument("--seq", default=False, action="store_true")
parser.add_argument("--hl", default=False, action="store_true")
parser.add_argument("--tf", default=False, action="store_true")
parser.add_argument("--stage", type=str, default="mid")
parser.add_argument("--comment", type=str, default="")
args = parser.parse_args()

if not any([args.seq, args.hl, args.tf]):
    raise ValueError("At least one of --seq, --hl, --tf should be set to True")

if args.track:
    os.environ["_USE_TRACK"] = "1"

if args.seq:
    os.environ["_USE_SEQ"] = "1"

if args.hl:
    os.environ["_USE_HALFLIFE"] = "1"

if args.tf:
    os.environ["_USE_TF"] = "1"

if args.comment:
    os.environ["RUN_COMMENT"] = args.comment

from models.BertDeepLncLocMultimodalExpression import (
    BertDeepLncLocMultimodalExpressionTrainer,
)
from models.TransformerSeqHalflifeToTF import TransformerSeqHalflifeToTFTrainer
from models.Conv2dDeepLncLocMultimodalExpression import (
    Conv2dDeepLncLocMultimodalExpressionTrainer,
)

MODEL_TRAINER_MAP = {
    "BertDeepLncLocMultimodalExpression": BertDeepLncLocMultimodalExpressionTrainer,
    "TransformerSeqHalflifeToTF": TransformerSeqHalflifeToTFTrainer,
    "Conv2dDeepLncLocMultimodalExpression": Conv2dDeepLncLocMultimodalExpressionTrainer,
}


if __name__ == "__main__":
    selected_model = MODEL_TRAINER_MAP.get(args.model)
    if selected_model:
        selected_model(model_kwargs={"stage": args.stage}).start()
        print(args)
    else:
        print("Model not found!")
