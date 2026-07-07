"""Точка входа: python -m projects.gazprom_emergency train --config ..."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(prog="gazprom_emergency")
    sub = parser.add_subparsers(dest="command")

    p_train = sub.add_parser("train", help="Обучить модель")
    p_train.add_argument("--config", required=True)

    p_predict = sub.add_parser("predict", help="Сделать предсказание")
    p_predict.add_argument("--config", required=True)
    p_predict.add_argument("--input", default=None)

    args = parser.parse_args()

    if args.command == "train":
        from .train import train
        train(args.config)
    elif args.command == "predict":
        from .predict import predict_batch
        predict_batch(args.config, args.input)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
