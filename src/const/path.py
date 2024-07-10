from pathlib import Path

# TODO
ROOT = Path(__file__).resolve().parents[1]
CONFIG_FOLDER = ROOT.joinpath("config")
DATA_FOLDER = ROOT.joinpath("data")
SRC_FOLDER = ROOT.joinpath("src")
TRAIN_FOLDER = ROOT.joinpath("train")
MODEL_FOLDER = SRC_FOLDER.joinpath("model")
