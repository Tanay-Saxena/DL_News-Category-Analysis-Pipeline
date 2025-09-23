"""
Clustering Pipeline Module
Implements clustering using KMeans and Agglomerative Clustering for news articles
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import pickle
import os
from datetime import datetime

class NewsClusteringPipeline:
    def __init__(self,
                 n_clusters_range: Tuple[int, int] = (5, 20),
                 random_state: int = 42):
        """
        Initialize clustering pipeline

        Args:
            n_clusters_range (Tuple[int, int]): Range of cluster numbers to try
            random_state (int): Random state for reproducibility
        """
        self.n_clusters_range = n_clusters_range
        self.random_state = random_state

        # Clustering results storage
        self.clustering_results = {}
        self.embeddings = None
        self.labels = None
        self.categories = None

        # Evaluation metrics
        self.evaluation_metrics = {}

        # Visualization data
        self.visualization_data = {}

    def prepare_embeddings(self,
                          embeddings_data: List[np.ndarray],
                          categories: List[str],
                          texts: List[str]) -> bool:
        """
        Prepare embeddings data for clustering

        Args:
            embeddings_data (List[np.ndarray]): List of embedding vectors
            categories (List[str]): Corresponding category labels
            texts (List[str]): Corresponding text descriptions

        Returns:
            bool: Success status
        """
        try:
            # Convert to numpy arrays
            self.embeddings = np.array(embeddings_data)
            self.categories = np.array(categories)
            self.texts = np.array(texts)

            # Standardize embeddings
            scaler = StandardScaler()
            self.embeddings_scaled = scaler.fit_transform(self.embeddings)

            print(f"Prepared {len(self.embeddings)} embeddings for clustering")
            print(f"Embedding dimension: {self.embeddings.shape[1]}")
            print(f"Unique categories: {len(np.unique(self.categories))}")

            return True

        except Exception as e:
            print(f"Error preparing embeddings: {e}")
            return False

    def find_optimal_clusters(self,
                            method: str = 'both',
                            max_clusters: int = 20) -> Dict[str, Any]:
        """
        Find optimal number of clusters using multiple methods

        Args:
            method (str): 'kmeans', 'agglomerative', or 'both'
            max_clusters (int): Maximum number of clusters to test

        Returns:
            Dict: Optimal clustering results
        """
        if self.embeddings is None:
            print("Error: No embeddings prepared. Call prepare_embeddings() first.")
            return {}

        print("Finding optimal number of clusters...")

        n_clusters_range = range(2, min(max_clusters + 1, len(self.embeddings) // 2))
        results = {}

        # Silhouette scores
        silhouette_scores = []

        for n_clusters in n_clusters_range:
            print(f"Testing {n_clusters} clusters...")

            cluster_results = {}

            # KMeans clustering
            if method in ['kmeans', 'both']:
                kmeans = KMeans(
                    n_clusters=n_clusters,
                    random_state=self.random_state,
                    n_init=10
                )
                kmeans_labels = kmeans.fit_predict(self.embeddings_scaled)

                # Calculate metrics
                silhouette_avg = silhouette_score(self.embeddings_scaled, kmeans_labels)
                ari_score = adjusted_rand_score(self.categories, kmeans_labels)
                nmi_score = normalized_mutual_info_score(self.categories, kmeans_labels)

                cluster_results['kmeans'] = {
                    'labels': kmeans_labels,
                    'silhouette_score': silhouette_avg,
                    'ari_score': ari_score,
                    'nmi_score': nmi_score,
                    'inertia': kmeans.inertia_
                }

                silhouette_scores.append(silhouette_avg)

            # Agglomerative clustering
            if method in ['agglomerative', 'both']:
                agg_clustering = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    linkage='ward'
                )
                agg_labels = agg_clustering.fit_predict(self.embeddings_scaled)

                # Calculate metrics
                silhouette_avg = silhouette_score(self.embeddings_scaled, agg_labels)
                ari_score = adjusted_rand_score(self.categories, agg_labels)
                nmi_score = normalized_mutual_info_score(self.categories, agg_labels)

                cluster_results['agglomerative'] = {
                    'labels': agg_labels,
                    'silhouette_score': silhouette_avg,
                    'ari_score': ari_score,
                    'nmi_score': nmi_score
                }

            results[n_clusters] = cluster_results

        # Find optimal number of clusters
        optimal_results = {}

        if method in ['kmeans', 'both']:
            kmeans_scores = [results[n]['kmeans']['silhouette_score'] for n in n_clusters_range]
            optimal_k = n_clusters_range[np.argmax(kmeans_scores)]
            optimal_results['kmeans'] = {
                'optimal_clusters': optimal_k,
                'silhouette_score': max(kmeans_scores),
                'results': results[optimal_k]['kmeans']
            }

        if method in ['agglomerative', 'both']:
            agg_scores = [results[n]['agglomerative']['silhouette_score'] for n in n_clusters_range]
            optimal_k = n_clusters_range[np.argmax(agg_scores)]
            optimal_results['agglomerative'] = {
                'optimal_clusters': optimal_k,
                'silhouette_score': max(agg_scores),
                'results': results[optimal_k]['agglomerative']
            }

        self.clustering_results = results
        self.evaluation_metrics = optimal_results

        return optimal_results

    def perform_clustering(self,
                          n_clusters: int,
                          method: str = 'kmeans') -> Dict[str, Any]:
        """
        Perform clustering with specified parameters

        Args:
            n_clusters (int): Number of clusters
            method (str): Clustering method ('kmeans' or 'agglomerative')

        Returns:
            Dict: Clustering results
        """
        if self.embeddings is None:
            print("Error: No embeddings prepared. Call prepare_embeddings() first.")
            return {}

        print(f"Performing {method} clustering with {n_clusters} clusters...")

        if method == 'kmeans':
            clusterer = KMeans(
                n_clusters=n_clusters,
                random_state=self.random_state,
                n_init=10
            )
            labels = clusterer.fit_predict(self.embeddings_scaled)

            result = {
                'method': 'kmeans',
                'n_clusters': n_clusters,
                'labels': labels,
                'cluster_centers': clusterer.cluster_centers_,
                'inertia': clusterer.inertia_,
                'silhouette_score': silhouette_score(self.embeddings_scaled, labels),
                'ari_score': adjusted_rand_score(self.categories, labels),
                'nmi_score': normalized_mutual_info_score(self.categories, labels)
            }

        elif method == 'agglomerative':
            clusterer = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage='ward'
            )
            labels = clusterer.fit_predict(self.embeddings_scaled)

            result = {
                'method': 'agglomerative',
                'n_clusters': n_clusters,
                'labels': labels,
                'silhouette_score': silhouette_score(self.embeddings_scaled, labels),
                'ari_score': adjusted_rand_score(self.categories, labels),
                'nmi_score': normalized_mutual_info_score(self.categories, labels)
            }

        else:
            print(f"Error: Unknown clustering method '{method}'")
            return {}

        self.labels = labels
        self.current_clustering = result

        print(f"Clustering completed. Silhouette score: {result['silhouette_score']:.3f}")
        print(f"ARI score: {result['ari_score']:.3f}, NMI score: {result['nmi_score']:.3f}")

        return result

    def analyze_clusters(self,
                        clustering_result: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Analyze clustering results and compare with original categories

        Args:
            clustering_result (Dict): Clustering result to analyze (uses current if None)

        Returns:
            Dict: Cluster analysis results
        """
        if clustering_result is None:
            clustering_result = self.current_clustering

        if clustering_result is None:
            print("Error: No clustering result to analyze")
            return {}

        labels = clustering_result['labels']

        # Create analysis DataFrame
        analysis_df = pd.DataFrame({
            'text': self.texts,
            'true_category': self.categories,
            'cluster': labels
        })

        # Cluster-category mapping
        cluster_category_mapping = {}
        cluster_analysis = {}

        for cluster_id in np.unique(labels):
            cluster_data = analysis_df[analysis_df['cluster'] == cluster_id]

            # Most common category in cluster
            most_common_category = cluster_data['true_category'].mode().iloc[0]
            category_counts = cluster_data['true_category'].value_counts()

            cluster_category_mapping[cluster_id] = most_common_category

            cluster_analysis[cluster_id] = {
                'size': len(cluster_data),
                'dominant_category': most_common_category,
                'category_distribution': category_counts.to_dict(),
                'purity': category_counts.iloc[0] / len(cluster_data),
                'sample_texts': cluster_data['text'].head(3).tolist()
            }

        # Overall analysis
        analysis_results = {
            'cluster_category_mapping': cluster_category_mapping,
            'cluster_analysis': cluster_analysis,
            'overall_purity': np.mean([info['purity'] for info in cluster_analysis.values()]),
            'total_clusters': len(np.unique(labels)),
            'clustering_metrics': {
                'silhouette_score': clustering_result['silhouette_score'],
                'ari_score': clustering_result['ari_score'],
                'nmi_score': clustering_result['nmi_score']
            }
        }

        return analysis_results

    def prepare_visualization_data(self,
                                  method: str = 'pca',
                                  n_components: int = 2) -> Dict[str, Any]:
        """
        Prepare data for visualization using dimensionality reduction

        Args:
            method (str): Reduction method ('pca' or 'tsne')
            n_components (int): Number of components for reduction

        Returns:
            Dict: Visualization data
        """
        if self.embeddings is None:
            print("Error: No embeddings prepared")
            return {}

        print(f"Preparing visualization data using {method.upper()}...")

        if method == 'pca':
            reducer = PCA(n_components=n_components, random_state=self.random_state)
            reduced_embeddings = reducer.fit_transform(self.embeddings_scaled)

            explained_variance = reducer.explained_variance_ratio_

        elif method == 'tsne':
            reducer = TSNE(n_components=n_components,
                          random_state=self.random_state,
                          perplexity=30,
                          n_iter=1000)
            reduced_embeddings = reducer.fit_transform(self.embeddings_scaled)
            explained_variance = None

        else:
            print(f"Error: Unknown reduction method '{method}'")
            return {}

        # Prepare visualization data
        viz_data = {
            'reduced_embeddings': reduced_embeddings,
            'categories': self.categories,
            'texts': self.texts,
            'method': method,
            'explained_variance': explained_variance
        }

        # Add clustering labels if available
        if hasattr(self, 'labels') and self.labels is not None:
            viz_data['cluster_labels'] = self.labels

        self.visualization_data = viz_data

        print(f"Visualization data prepared. Shape: {reduced_embeddings.shape}")
        if explained_variance is not None:
            print(f"Explained variance: {explained_variance}")

        return viz_data

    def compare_with_original_labels(self,
                                   clustering_result: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Compare clustering results with original category labels

        Args:
            clustering_result (Dict): Clustering result to compare

        Returns:
            Dict: Comparison results
        """
        if clustering_result is None:
            clustering_result = self.current_clustering

        if clustering_result is None:
            print("Error: No clustering result to compare")
            return {}

        labels = clustering_result['labels']

        # Create comparison DataFrame
        comparison_df = pd.DataFrame({
            'true_category': self.categories,
            'predicted_cluster': labels
        })

        # Calculate confusion matrix
        confusion_matrix = pd.crosstab(
            comparison_df['true_category'],
            comparison_df['predicted_cluster'],
            margins=True
        )

        # Find best cluster-category matches
        cluster_category_matches = {}
        for cluster in np.unique(labels):
            cluster_data = comparison_df[comparison_df['predicted_cluster'] == cluster]
            most_common_category = cluster_data['true_category'].mode().iloc[0]
            match_count = (cluster_data['true_category'] == most_common_category).sum()
            total_count = len(cluster_data)

            cluster_category_matches[cluster] = {
                'best_match_category': most_common_category,
                'match_count': match_count,
                'total_count': total_count,
                'accuracy': match_count / total_count
            }

        # Overall accuracy
        correct_predictions = 0
        for cluster, match_info in cluster_category_matches.items():
            correct_predictions += match_info['match_count']

        overall_accuracy = correct_predictions / len(labels)

        comparison_results = {
            'confusion_matrix': confusion_matrix,
            'cluster_category_matches': cluster_category_matches,
            'overall_accuracy': overall_accuracy,
            'clustering_metrics': {
                'ari_score': clustering_result['ari_score'],
                'nmi_score': clustering_result['nmi_score'],
                'silhouette_score': clustering_result['silhouette_score']
            }
        }

        return comparison_results

    def save_clustering_results(self,
                               filename: str = "clustering_results.pkl") -> bool:
        """
        Save clustering results to file

        Args:
            filename (str): Output filename

        Returns:
            bool: Success status
        """
        try:
            results_to_save = {
                'clustering_results': self.clustering_results,
                'evaluation_metrics': self.evaluation_metrics,
                'current_clustering': getattr(self, 'current_clustering', None),
                'embeddings': self.embeddings,
                'categories': self.categories,
                'texts': self.texts,
                'labels': getattr(self, 'labels', None),
                'visualization_data': self.visualization_data,
                'timestamp': datetime.now().isoformat()
            }

            with open(filename, 'wb') as f:
                pickle.dump(results_to_save, f)

            print(f"Clustering results saved to {filename}")
            return True

        except Exception as e:
            print(f"Error saving clustering results: {e}")
            return False

    def load_clustering_results(self,
                               filename: str = "clustering_results.pkl") -> bool:
        """
        Load clustering results from file

        Args:
            filename (str): Input filename

        Returns:
            bool: Success status
        """
        try:
            with open(filename, 'rb') as f:
                results = pickle.load(f)

            self.clustering_results = results.get('clustering_results', {})
            self.evaluation_metrics = results.get('evaluation_metrics', {})
            self.current_clustering = results.get('current_clustering', None)
            self.embeddings = results.get('embeddings', None)
            self.categories = results.get('categories', None)
            self.texts = results.get('texts', None)
            self.labels = results.get('labels', None)
            self.visualization_data = results.get('visualization_data', {})

            print(f"Clustering results loaded from {filename}")
            return True

        except Exception as e:
            print(f"Error loading clustering results: {e}")
            return False

def main():
    """Main function to demonstrate clustering pipeline"""
    print("NewsClusteringPipeline class created successfully!")
    print("This class provides comprehensive clustering analysis for news articles.")

    # Example usage:
    """
    # Initialize pipeline
    pipeline = NewsClusteringPipeline()

    # Prepare embeddings (assuming you have embeddings data)
    # pipeline.prepare_embeddings(embeddings, categories, texts)

    # Find optimal clusters
    # optimal_results = pipeline.find_optimal_clusters()

    # Perform clustering
    # clustering_result = pipeline.perform_clustering(n_clusters=10)

    # Analyze results
    # analysis = pipeline.analyze_clusters()
    """

if __name__ == "__main__":
    main()
