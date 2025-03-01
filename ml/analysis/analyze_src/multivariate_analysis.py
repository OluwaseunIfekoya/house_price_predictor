from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Abstract Base Class for Multivariate Analysis
# --------------------------------------
# This class defines a template for performaing multivariate analysis. 
# Subclasses can override sepcific steps like correlation heatma and pair plot geenration. 
class MultivariateAnalysisTemplate(ABC):
    def analyze(self, df: pd.DataFrame):
        """
        Perform a comprehensive multivariate analysis by generating a correlation heatmap and pair plot. 

        Parameters: 
        df (pd.DataFrame): The dataframe containing the data to be anakyzed.

        Returns:
        None: This method orchestrates the multivariate analysis process.
        """
        self.generate_correlation_heatmap(df)
        self.generate_pairplot(df)

    @abstractmethod
    def generate_correlation_heatmap(self, df: pd.DataFrame):
        """
        Generate and display a heatmap of the correlations between features. 

        Returns:
        None: This method should generate and displat a correlation heatmap.
        """
        pass

    @abstractmethod
    def generate_pairplot(self, df: pd.DataFrame):
        """
        Generate and display a pair plot of the selected features. 

        Returns:
        None: This method should generate and display a pair plot. 
        """
        pass

# Concrete class for Multivariate Analysis with Correlation Heatmap and Pair Plot. 
# -----------------------------
# This class implements the methods to generate a correlation heatmap and a pair plot. 
class SimpleMultivariateAnalysis(MultivariateAnalysisTemplate):
    def generate_correlation_heatmap(self, df: pd.DataFrame):
        """
        Genarates and displays a correlation heatmap for the numerical features in the dataframe. 

        Returns:
        None: Displays a heatmap showing correlations between numerical features. 
        """

        plt.figure(figsize=(12, 10))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
        plt.title("Correlation Heatmap")
        plt.show()

    def generate_pairplot(self, df: pd.DataFrame):
        """
        Generates and displays a pair plot for the selected features in the dataframe. 

        Returns:
        None: Displays a pair plot for the selected features. 
        """

        sns.pairplot(df)
        plt.suptitle("Pair Plot of Selected Features", y=1.02)
        plt.show()

