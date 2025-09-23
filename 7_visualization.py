"""
Visualization Module
Creates comprehensive visualizations for news category classification results
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class NewsClassificationVisualizer:
    def __init__(self,
                 output_dir: str = "visualizations",
                 figure_size: Tuple[int, int] = (12, 8),
                 dpi: int = 300):
        """
        Initialize NewsClassificationVisualizer

        Args:
            output_dir (str): Directory to save visualizations
            figure_size (Tuple[int, int]): Default figure size
            dpi (int): DPI for saved figures
        """
        self.output_dir = output_dir
        self.figure_size = figure_size
        self.dpi = dpi

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Set matplotlib parameters
        plt.rcParams['figure.figsize'] = figure_size
        plt.rcParams['savefig.dpi'] = dpi
        plt.rcParams['font.size'] = 10

    def plot_category_distribution(self,
                                 df: pd.DataFrame,
                                 category_column: str = 'category',
                                 title: str = "Distribution of News Categories",
                                 save: bool = True) -> plt.Figure:
        """
        Plot distribution of categories in the dataset

        Args:
            df (pd.DataFrame): DataFrame with category data
            category_column (str): Name of category column
            title (str): Plot title
            save (bool): Whether to save the plot

        Returns:
            plt.Figure: Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figure_size)

        # Get category counts
        category_counts = df[category_column].value_counts()

        # Create bar plot
        bars = ax.bar(range(len(category_counts)), category_counts.values,
                     color=plt.cm.Set3(np.linspace(0, 1, len(category_counts))))

        # Customize plot
        ax.set_xlabel('Categories')
        ax.set_ylabel('Number of Articles')
        ax.set_title(title)
        ax.set_xticks(range(len(category_counts)))
        ax.set_xticklabels(category_counts.index, rotation=45, ha='right')

        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{int(height)}', ha='center', va='bottom')

        plt.tight_layout()

        if save:
            filename = os.path.join(self.output_dir, "category_distribution.png")
            plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
            print(f"Category distribution plot saved to {filename}")

        return fig

    def plot_clustering_results(self,
                               reduced_embeddings: np.ndarray,
                               labels: np.ndarray,
                               categories: np.ndarray,
                               method: str = 'PCA',
                               title: str = "Clustering Results Visualization",
                               save: bool = True) -> plt.Figure:
        """
        Plot clustering results using dimensionality reduction

        Args:
            reduced_embeddings (np.ndarray): 2D reduced embeddings
            labels (np.ndarray): Cluster labels
            categories (np.ndarray): True category labels
            method (str): Dimensionality reduction method used
            title (str): Plot title
            save (bool): Whether to save the plot

        Returns:
            plt.Figure: Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Clusters
        scatter1 = ax1.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1],
                              c=labels, cmap='tab20', alpha=0.7, s=50)
        ax1.set_xlabel(f'{method} Component 1')
        ax1.set_ylabel(f'{method} Component 2')
        ax1.set_title('Clustering Results')
        ax1.grid(True, alpha=0.3)

        # Add colorbar for clusters
        cbar1 = plt.colorbar(scatter1, ax=ax1)
        cbar1.set_label('Cluster ID')

        # Plot 2: True categories
        unique_categories = np.unique(categories)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_categories)))
        category_colors = {cat: colors[i] for i, cat in enumerate(unique_categories)}

        for category in unique_categories:
            mask = categories == category
            ax2.scatter(reduced_embeddings[mask, 0], reduced_embeddings[mask, 1],
                       c=[category_colors[category]], label=category, alpha=0.7, s=50)

        ax2.set_xlabel(f'{method} Component 1')
        ax2.set_ylabel(f'{method} Component 2')
        ax2.set_title('True Categories')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()

        if save:
            filename = os.path.join(self.output_dir, "clustering_visualization.png")
            plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
            print(f"Clustering visualization saved to {filename}")

        return fig

    def plot_confusion_matrix(self,
                            confusion_matrix: pd.DataFrame,
                            title: str = "Confusion Matrix",
                            normalize: bool = True,
                            save: bool = True) -> plt.Figure:
        """
        Plot confusion matrix

        Args:
            confusion_matrix (pd.DataFrame): Confusion matrix data
            title (str): Plot title
            normalize (bool): Whether to normalize the matrix
            save (bool): Whether to save the plot

        Returns:
            plt.Figure: Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Normalize if requested
        if normalize:
            cm_normalized = confusion_matrix.div(confusion_matrix.sum(axis=1), axis=0)
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=ax)
            ax.set_title(f"{title} (Normalized)")
        else:
            sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(title)

        ax.set_xlabel('Predicted Category')
        ax.set_ylabel('True Category')

        plt.tight_layout()

        if save:
            filename = os.path.join(self.output_dir, "confusion_matrix.png")
            plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
            print(f"Confusion matrix saved to {filename}")

        return fig

    def plot_accuracy_by_category(self,
                                results_df: pd.DataFrame,
                                title: str = "Accuracy by Category",
                                save: bool = True) -> plt.Figure:
        """
        Plot accuracy for each category

        Args:
            results_df (pd.DataFrame): Results DataFrame
            title (str): Plot title
            save (bool): Whether to save the plot

        Returns:
            plt.Figure: Matplotlib figure
        """
        # Calculate accuracy by category
        category_accuracy = results_df.groupby('ground_truth_category')['is_match'].agg(['count', 'sum', 'mean'])
        category_accuracy.columns = ['total', 'correct', 'accuracy']
        category_accuracy = category_accuracy.sort_values('accuracy', ascending=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Accuracy bars
        bars = ax1.barh(range(len(category_accuracy)), category_accuracy['accuracy'],
                       color=plt.cm.viridis(np.linspace(0, 1, len(category_accuracy))))
        ax1.set_yticks(range(len(category_accuracy)))
        ax1.set_yticklabels(category_accuracy.index)
        ax1.set_xlabel('Accuracy')
        ax1.set_title('Accuracy by Category')
        ax1.grid(True, alpha=0.3)

        # Add accuracy values on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax1.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}', ha='left', va='center')

        # Plot 2: Sample size
        ax2.barh(range(len(category_accuracy)), category_accuracy['total'],
                color=plt.cm.plasma(np.linspace(0, 1, len(category_accuracy))))
        ax2.set_yticks(range(len(category_accuracy)))
        ax2.set_yticklabels(category_accuracy.index)
        ax2.set_xlabel('Number of Samples')
        ax2.set_title('Sample Size by Category')
        ax2.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()

        if save:
            filename = os.path.join(self.output_dir, "accuracy_by_category.png")
            plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
            print(f"Accuracy by category plot saved to {filename}")

        return fig

    def plot_similarity_analysis(self,
                               results_df: pd.DataFrame,
                               title: str = "Similarity Score Analysis",
                               save: bool = True) -> plt.Figure:
        """
        Plot similarity score analysis

        Args:
            results_df (pd.DataFrame): Results DataFrame
            title (str): Plot title
            save (bool): Whether to save the plot

        Returns:
            plt.Figure: Matplotlib figure
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: Similarity distribution
        ax1.hist(results_df['similarity_score'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(results_df['similarity_score'].mean(), color='red', linestyle='--',
                   label=f'Mean: {results_df["similarity_score"].mean():.3f}')
        ax1.set_xlabel('Similarity Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Similarity Scores')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Similarity by match status
        match_data = [results_df[results_df['is_match']]['similarity_score'].values,
                     results_df[~results_df['is_match']]['similarity_score'].values]
        ax2.boxplot(match_data, labels=['Correct', 'Incorrect'])
        ax2.set_ylabel('Similarity Score')
        ax2.set_title('Similarity Scores by Prediction Accuracy')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Similarity vs Confidence
        scatter = ax3.scatter(results_df['similarity_score'], results_df['deepseek_confidence'],
                             c=results_df['is_match'], cmap='RdYlGn', alpha=0.6)
        ax3.set_xlabel('Similarity Score')
        ax3.set_ylabel('DeepSeek Confidence')
        ax3.set_title('Similarity vs Confidence')
        ax3.grid(True, alpha=0.3)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('Correct Prediction')

        # Plot 4: Similarity by category (top 10)
        top_categories = results_df['ground_truth_category'].value_counts().head(10).index
        category_similarity = results_df[results_df['ground_truth_category'].isin(top_categories)]

        sns.boxplot(data=category_similarity, x='ground_truth_category', y='similarity_score', ax=ax4)
        ax4.set_xlabel('Category')
        ax4.set_ylabel('Similarity Score')
        ax4.set_title('Similarity Scores by Category (Top 10)')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()

        if save:
            filename = os.path.join(self.output_dir, "similarity_analysis.png")
            plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
            print(f"Similarity analysis plot saved to {filename}")

        return fig

    def plot_error_analysis(self,
                           results_df: pd.DataFrame,
                           title: str = "Error Analysis",
                           save: bool = True) -> plt.Figure:
        """
        Plot error analysis and patterns

        Args:
            results_df (pd.DataFrame): Results DataFrame
            title (str): Plot title
            save (bool): Whether to save the plot

        Returns:
            plt.Figure: Matplotlib figure
        """
        # Get error patterns
        error_df = results_df[~results_df['is_match']]
        error_patterns = error_df.groupby(['ground_truth_category', 'predicted_category']).size().reset_index(name='count')
        error_patterns = error_patterns.sort_values('count', ascending=False).head(15)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: Top error patterns
        error_labels = [f"{row['ground_truth_category']} → {row['predicted_category']}"
                       for _, row in error_patterns.iterrows()]
        bars = ax1.barh(range(len(error_patterns)), error_patterns['count'],
                       color=plt.cm.Reds(np.linspace(0.3, 1, len(error_patterns))))
        ax1.set_yticks(range(len(error_patterns)))
        ax1.set_yticklabels(error_labels)
        ax1.set_xlabel('Number of Errors')
        ax1.set_title('Top Error Patterns')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Error rate by category
        category_errors = results_df.groupby('ground_truth_category').agg({
            'is_match': ['count', 'sum']
        })
        category_errors.columns = ['total', 'correct']
        category_errors['error_rate'] = 1 - (category_errors['correct'] / category_errors['total'])
        category_errors = category_errors.sort_values('error_rate', ascending=True).head(15)

        bars = ax2.barh(range(len(category_errors)), category_errors['error_rate'],
                       color=plt.cm.Oranges(np.linspace(0.3, 1, len(category_errors))))
        ax2.set_yticks(range(len(category_errors)))
        ax2.set_yticklabels(category_errors.index)
        ax2.set_xlabel('Error Rate')
        ax2.set_title('Error Rate by Category (Top 15)')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Text length vs accuracy
        length_bins = pd.cut(results_df['text_length'], bins=10)
        length_accuracy = results_df.groupby(length_bins)['is_match'].mean()

        ax3.plot(range(len(length_accuracy)), length_accuracy.values, marker='o', linewidth=2)
        ax3.set_xlabel('Text Length Bins')
        ax3.set_ylabel('Accuracy')
        ax3.set_title('Accuracy by Text Length')
        ax3.grid(True, alpha=0.3)
        ax3.set_xticks(range(len(length_accuracy)))
        ax3.set_xticklabels([str(interval) for interval in length_accuracy.index], rotation=45)

        # Plot 4: Confidence distribution for correct vs incorrect
        correct_conf = results_df[results_df['is_match']]['deepseek_confidence']
        incorrect_conf = results_df[~results_df['is_match']]['deepseek_confidence']

        ax4.hist(correct_conf, bins=20, alpha=0.7, label='Correct', color='green')
        ax4.hist(incorrect_conf, bins=20, alpha=0.7, label='Incorrect', color='red')
        ax4.set_xlabel('DeepSeek Confidence')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Confidence Distribution by Accuracy')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()

        if save:
            filename = os.path.join(self.output_dir, "error_analysis.png")
            plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
            print(f"Error analysis plot saved to {filename}")

        return fig

    def create_interactive_dashboard(self,
                                   results_df: pd.DataFrame,
                                   reduced_embeddings: Optional[np.ndarray] = None,
                                   labels: Optional[np.ndarray] = None) -> go.Figure:
        """
        Create an interactive Plotly dashboard

        Args:
            results_df (pd.DataFrame): Results DataFrame
            reduced_embeddings (np.ndarray): 2D reduced embeddings
            labels (np.ndarray): Cluster labels

        Returns:
            go.Figure: Interactive Plotly figure
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Accuracy by Category', 'Similarity vs Confidence',
                          'Text Length Distribution', 'Confidence Distribution'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "histogram"}, {"type": "histogram"}]]
        )

        # Plot 1: Accuracy by category
        category_accuracy = results_df.groupby('ground_truth_category')['is_match'].mean().sort_values(ascending=True)
        fig.add_trace(
            go.Bar(x=category_accuracy.values, y=category_accuracy.index,
                  orientation='h', name='Accuracy', marker_color='lightblue'),
            row=1, col=1
        )

        # Plot 2: Similarity vs Confidence
        fig.add_trace(
            go.Scatter(x=results_df['similarity_score'],
                      y=results_df['deepseek_confidence'],
                      mode='markers',
                      marker=dict(color=results_df['is_match'],
                                colorscale='RdYlGn',
                                size=8,
                                opacity=0.6),
                      text=results_df['ground_truth_category'],
                      name='Predictions'),
            row=1, col=2
        )

        # Plot 3: Text length distribution
        fig.add_trace(
            go.Histogram(x=results_df['text_length'],
                        nbinsx=30, name='Text Length', marker_color='lightgreen'),
            row=2, col=1
        )

        # Plot 4: Confidence distribution
        fig.add_trace(
            go.Histogram(x=results_df['deepseek_confidence'],
                        nbinsx=30, name='Confidence', marker_color='lightcoral'),
            row=2, col=2
        )

        # Update layout
        fig.update_layout(
            title_text="News Classification Dashboard",
            showlegend=False,
            height=800,
            width=1200
        )

        # Save interactive plot
        filename = os.path.join(self.output_dir, "interactive_dashboard.html")
        fig.write_html(filename)
        print(f"Interactive dashboard saved to {filename}")

        return fig

    def generate_all_visualizations(self,
                                  results_df: pd.DataFrame,
                                  confusion_matrix: Optional[pd.DataFrame] = None,
                                  reduced_embeddings: Optional[np.ndarray] = None,
                                  labels: Optional[np.ndarray] = None,
                                  categories: Optional[np.ndarray] = None) -> Dict[str, plt.Figure]:
        """
        Generate all visualizations

        Args:
            results_df (pd.DataFrame): Results DataFrame
            confusion_matrix (pd.DataFrame): Confusion matrix
            reduced_embeddings (np.ndarray): 2D reduced embeddings
            labels (np.ndarray): Cluster labels
            categories (np.ndarray): True category labels

        Returns:
            Dict[str, plt.Figure]: Dictionary of generated figures
        """
        figures = {}

        print("Generating all visualizations...")

        # 1. Category distribution
        if 'ground_truth_category' in results_df.columns:
            figures['category_distribution'] = self.plot_category_distribution(
                results_df, 'ground_truth_category'
            )

        # 2. Confusion matrix
        if confusion_matrix is not None:
            figures['confusion_matrix'] = self.plot_confusion_matrix(confusion_matrix)

        # 3. Accuracy by category
        figures['accuracy_by_category'] = self.plot_accuracy_by_category(results_df)

        # 4. Similarity analysis
        figures['similarity_analysis'] = self.plot_similarity_analysis(results_df)

        # 5. Error analysis
        figures['error_analysis'] = self.plot_error_analysis(results_df)

        # 6. Clustering visualization
        if reduced_embeddings is not None and labels is not None and categories is not None:
            figures['clustering_results'] = self.plot_clustering_results(
                reduced_embeddings, labels, categories
            )

        # 7. Interactive dashboard
        figures['interactive_dashboard'] = self.create_interactive_dashboard(
            results_df, reduced_embeddings, labels
        )

        print(f"Generated {len(figures)} visualizations")
        return figures

def main():
    """Main function to demonstrate visualization capabilities"""
    print("NewsClassificationVisualizer class created successfully!")
    print("This class provides comprehensive visualization capabilities for news classification results.")

    # Example usage:
    """
    # Initialize visualizer
    visualizer = NewsClassificationVisualizer()

    # Generate individual plots
    # fig1 = visualizer.plot_category_distribution(df)
    # fig2 = visualizer.plot_confusion_matrix(confusion_matrix)

    # Generate all visualizations
    # figures = visualizer.generate_all_visualizations(results_df, confusion_matrix)
    """

if __name__ == "__main__":
    main()
