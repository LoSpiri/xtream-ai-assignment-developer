from pathlib import Path
from typing import List, Tuple
import pandas as pd
import plotly.express as px
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt


class ExplorationUtils:

    @staticmethod
    def scatter_matrix_plot(df: pd.DataFrame, figsize: Tuple[int, int], path: Path):
        scatter_matrix(df.select_dtypes(include=["number"]), figsize=figsize)
        plt.savefig(path)
        plt.close()

    @staticmethod
    def hist_plot(df: pd.DataFrame, bins: int, figsize: Tuple[int, int], path: Path):
        df.hist(bins=bins, figsize=figsize)
        plt.savefig(path)
        plt.close()

    @staticmethod
    def violin_plot_by_price(df: pd.DataFrame, column: str, path: Path):
        px.violin(df, x=column, y='price', color=column, title=f'Price by {column}')
        plt.savefig(path)
        plt.close()

    @staticmethod
    def scatter_plot_by_price_vs_carat(df: pd.DataFrame, column: str, path: Path):
        px.scatter(df, x='carat', y='price', color=column, title=f'Price vs carat with {column}')
        plt.savefig(path)
        plt.close()
