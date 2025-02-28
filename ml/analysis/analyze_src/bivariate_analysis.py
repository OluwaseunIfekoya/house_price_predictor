from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Abstract Base Class for Bivariate Analysis Strategy
# -------------------------
# This class defines a common interface for bivariate analysis strategies. 
# Subclasses must implement the analyze method.
class BivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Perform bivariate analysis on two features of the DataFrame. 

        Parameters: 
        df (pd.DataFrame): The dataframe containing the data. 
        feature1 (str): The name of the first feature to be analyzed.
        feature2 (str): The name of the second feature to be analyzed. 

        Returns:
        None: This method visualizes the relationship between the two features. 

        """
        pass

# Concrete Strategy for Numerical vs Numerical analysis

class NumericalVsNumericalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Plots the relationship between two numerical features using a scatter plot. 

        Returns:
        None: Displays a scatter plot showing the relationship between the two features. 
        """

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=feature1, y=feature2, data=df)
        plt.title(f"{feature1} vs {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()

class CategoricalVsNumericalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Plots the relationship between a categorical feature and a numerical feature using a boxplot. 


        Returns:
        None: Displays a box plot showing  the relationship between the categorical feature and numerical feature. 
        """

        plt.figure(figsize=(10,6))
        sns.boxplot(x=feature1, y=feature2, data=df)
        plt.title(f"{feature1} vs {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.xticks(rotation=45)
        plt.show()

# Context Class that uses a Bivariate Analysis Strategy
# ---------------------------------------
# This class allows you to switch between different bivariate analysis strategies. 
class BivariateAnalyzer:
    def __init__(self, strategy: BivariateAnalysisStrategy):
        """
        Initializes the BivariateAnalyzer with a specific analysis strategy. 

        Parameters: 
        strategy (BivariateAnalysisStrategy): The strategy to be used for bivariate analysis. 

        Returns:
        None
        """
        self._strategy = strategy
    
    def set_strategy(self, strategy: BivariateAnalysisStrategy):
        """
        Sets a new strategy for the BivariateAnalyzer. 

        Parameters: 
        strategy (BivariateAnalysisStrategy): The new strategy to be used for bivariate analysis.

        Returns:
        None
        """

        self._strategy = strategy 

    def execute_analysis(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Executes the bivariate analysis using the current strategy. 

        Returns:
        None: Executes the strategy's analysis method and visualizes the results/
        """
        self._strategy.analyze(df, feature1, feature2)