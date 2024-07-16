from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_FOLDER = ROOT.joinpath("config")
DATA_FOLDER = ROOT.joinpath("data")
SRC_FOLDER = ROOT.joinpath("src")
TRAIN_FOLDER = ROOT.joinpath("train")
MODEL_FOLDER = SRC_FOLDER.joinpath("model")
DB_PATH = ROOT.joinpath("interactions.db")
