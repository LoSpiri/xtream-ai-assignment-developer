import plotly.express as px


class ExplorationUtils:

    @staticmethod
    def plot_diamonds_price_by(df, column):
        return px.violin(df, x=column, y='price', color=column, title=f'Price by {column}')

    @staticmethod
    def scatter_diamods_by(df, column):
        return px.scatter(df, x='carat', y='price', color=column, title=f'Price vs carat with {column}')
