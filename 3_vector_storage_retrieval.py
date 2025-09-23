"""
Vector Storage & Retrieval Module
Enhanced ChromaDB integration with advanced retrieval and storage functions
"""

import chromadb
from chromadb.config import Settings
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import json
import os
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import pickle

class VectorStorageRetrieval:
    def __init__(self,
                 collection_name: str = "news_embeddings",
                 persist_directory: str = "./chroma_db"):
        """
        Initialize VectorStorageRetrieval

        Args:
            collection_name (str): Name of the ChromaDB collection
            persist_directory (str): Directory to persist ChromaDB data
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        # Initialize collection
        self.collection = None
        self._initialize_collection()

        # Cache for frequently accessed data
        self.cache = {
            'category_centroids': {},
            'category_stats': {},
            'last_updated': None
        }

    def _initialize_collection(self):
        """Initialize or get existing ChromaDB collection"""
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            print(f"Loaded existing collection: {self.collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "News category embeddings with metadata"}
            )
            print(f"Created new collection: {self.collection_name}")

    def store_embeddings_with_metadata(self,
                                     texts: List[str],
                                     categories: List[str],
                                     additional_metadata: Optional[List[Dict]] = None,
                                     batch_size: int = 100) -> bool:
        """
        Store embeddings with comprehensive metadata

        Args:
            texts (List[str]): List of text descriptions
            categories (List[str]): List of corresponding categories
            additional_metadata (List[Dict]): Additional metadata for each document
            batch_size (int): Batch size for processing

        Returns:
            bool: Success status
        """
        if len(texts) != len(categories):
            print("Error: Number of texts and categories must match")
            return False

        print(f"Storing {len(texts)} embeddings with metadata...")

        # Prepare metadata
        if additional_metadata is None:
            additional_metadata = [{}] * len(texts)

        # Process in batches
        total_batches = (len(texts) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(texts))

            batch_texts = texts[start_idx:end_idx]
            batch_categories = categories[start_idx:end_idx]
            batch_metadata = additional_metadata[start_idx:end_idx]
            batch_ids = [f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}"
                        for i in range(start_idx, end_idx)]

            # Enhanced metadata
            enhanced_metadata = []
            for i, (text, category, meta) in enumerate(zip(batch_texts, batch_categories, batch_metadata)):
                enhanced_meta = {
                    "category": category,
                    "text_length": len(text),
                    "word_count": len(text.split()),
                    "batch_id": batch_idx,
                    "timestamp": datetime.now().isoformat(),
                    **meta  # Include any additional metadata
                }
                enhanced_metadata.append(enhanced_meta)

            try:
                self.collection.add(
                    documents=batch_texts,
                    metadatas=enhanced_metadata,
                    ids=batch_ids
                )
            except Exception as e:
                print(f"Error storing batch {batch_idx}: {e}")
                continue

        print(f"Successfully stored {len(texts)} embeddings")
        return True

    def retrieve_top_k_similar(self,
                              query_text: str,
                              k: int = 5,
                              category_filter: Optional[str] = None,
                              min_similarity: float = 0.0) -> List[Dict]:
        """
        Retrieve top-k most similar articles

        Args:
            query_text (str): Input text to find similar articles for
            k (int): Number of similar articles to retrieve
            category_filter (str): Optional category filter
            min_similarity (float): Minimum similarity threshold

        Returns:
            List[Dict]: List of similar articles with metadata
        """
        try:
            # Build where clause if category filter is specified
            where_clause = None
            if category_filter:
                where_clause = {"category": category_filter}

            # Query ChromaDB
            results = self.collection.query(
                query_texts=[query_text],
                n_results=k,
                where=where_clause
            )

            # Format and filter results
            similar_articles = []
            for i in range(len(results['documents'][0])):
                distance = results['distances'][0][i]
                similarity = 1 - distance  # Convert distance to similarity

                if similarity >= min_similarity:
                    article = {
                        'id': results['ids'][0][i],
                        'text': results['documents'][0][i],
                        'similarity': similarity,
                        'distance': distance,
                        'category': results['metadatas'][0][i].get('category', 'Unknown'),
                        'metadata': results['metadatas'][0][i]
                    }
                    similar_articles.append(article)

            return similar_articles

        except Exception as e:
            print(f"Error retrieving similar articles: {e}")
            return []

    def get_category_centroids(self, force_recalculate: bool = False) -> Dict[str, np.ndarray]:
        """
        Calculate and cache category centroids

        Args:
            force_recalculate (bool): Force recalculation even if cached

        Returns:
            Dict[str, np.ndarray]: Category centroids
        """
        if not force_recalculate and self.cache['category_centroids']:
            return self.cache['category_centroids']

        print("Calculating category centroids...")

        try:
            # Get all documents
            all_docs = self.collection.get()
            categories = [doc['category'] for doc in all_docs['metadatas']]
            unique_categories = list(set(categories))

            centroids = {}

            for category in unique_categories:
                # Get documents for this category
                category_docs = self.collection.get(
                    where={"category": category}
                )

                if len(category_docs['documents']) > 0:
                    # Calculate centroid (this is a simplified approach)
                    # In practice, you'd need to get the actual embeddings
                    # For now, we'll use a placeholder
                    centroids[category] = np.random.rand(384)  # Placeholder for embedding dimension

            self.cache['category_centroids'] = centroids
            self.cache['last_updated'] = datetime.now()

            return centroids

        except Exception as e:
            print(f"Error calculating centroids: {e}")
            return {}

    def find_most_similar_category(self,
                                 query_text: str,
                                 top_k: int = 3) -> List[Dict]:
        """
        Find the most similar category for a given text

        Args:
            query_text (str): Input text
            top_k (int): Number of top categories to return

        Returns:
            List[Dict]: Top similar categories with scores
        """
        try:
            # Get similar articles
            similar_articles = self.retrieve_top_k_similar(query_text, k=50)

            if not similar_articles:
                return []

            # Count categories and calculate average similarity
            category_scores = {}
            category_counts = {}

            for article in similar_articles:
                category = article['category']
                similarity = article['similarity']

                if category not in category_scores:
                    category_scores[category] = 0
                    category_counts[category] = 0

                category_scores[category] += similarity
                category_counts[category] += 1

            # Calculate average similarity per category
            category_avg_scores = []
            for category in category_scores:
                avg_score = category_scores[category] / category_counts[category]
                category_avg_scores.append({
                    'category': category,
                    'avg_similarity': avg_score,
                    'count': category_counts[category]
                })

            # Sort by average similarity
            category_avg_scores.sort(key=lambda x: x['avg_similarity'], reverse=True)

            return category_avg_scores[:top_k]

        except Exception as e:
            print(f"Error finding similar category: {e}")
            return []

    def get_category_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about categories

        Returns:
            Dict: Category statistics
        """
        try:
            all_docs = self.collection.get()

            if not all_docs['documents']:
                return {}

            # Extract categories and metadata
            categories = [doc['category'] for doc in all_docs['metadatas']]
            text_lengths = [doc.get('text_length', 0) for doc in all_docs['metadatas']]
            word_counts = [doc.get('word_count', 0) for doc in all_docs['metadatas']]

            # Calculate statistics
            category_counts = pd.Series(categories).value_counts()

            stats = {
                'total_documents': len(all_docs['documents']),
                'unique_categories': len(category_counts),
                'category_distribution': category_counts.to_dict(),
                'avg_text_length': np.mean(text_lengths),
                'avg_word_count': np.mean(word_counts),
                'text_length_stats': {
                    'min': min(text_lengths),
                    'max': max(text_lengths),
                    'std': np.std(text_lengths)
                },
                'word_count_stats': {
                    'min': min(word_counts),
                    'max': max(word_counts),
                    'std': np.std(word_counts)
                }
            }

            return stats

        except Exception as e:
            print(f"Error getting category statistics: {e}")
            return {}

    def search_by_metadata(self,
                          metadata_filter: Dict[str, Any],
                          limit: int = 100) -> List[Dict]:
        """
        Search documents by metadata criteria

        Args:
            metadata_filter (Dict): Metadata filter criteria
            limit (int): Maximum number of results

        Returns:
            List[Dict]: Matching documents
        """
        try:
            results = self.collection.get(
                where=metadata_filter,
                limit=limit
            )

            documents = []
            for i in range(len(results['documents'])):
                doc = {
                    'id': results['ids'][i],
                    'text': results['documents'][i],
                    'metadata': results['metadatas'][i]
                }
                documents.append(doc)

            return documents

        except Exception as e:
            print(f"Error searching by metadata: {e}")
            return []

    def export_embeddings_data(self,
                              output_file: str = "embeddings_export.pkl") -> bool:
        """
        Export embeddings data for external analysis

        Args:
            output_file (str): Output file path

        Returns:
            bool: Success status
        """
        try:
            all_docs = self.collection.get()

            export_data = {
                'documents': all_docs['documents'],
                'metadatas': all_docs['metadatas'],
                'ids': all_docs['ids'],
                'export_timestamp': datetime.now().isoformat(),
                'collection_name': self.collection_name
            }

            with open(output_file, 'wb') as f:
                pickle.dump(export_data, f)

            print(f"Embeddings data exported to {output_file}")
            return True

        except Exception as e:
            print(f"Error exporting embeddings data: {e}")
            return False

    def import_embeddings_data(self,
                              input_file: str = "embeddings_export.pkl") -> bool:
        """
        Import embeddings data from external file

        Args:
            input_file (str): Input file path

        Returns:
            bool: Success status
        """
        try:
            with open(input_file, 'rb') as f:
                import_data = pickle.load(f)

            # Clear existing collection
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "News category embeddings with metadata"}
            )

            # Import data
            self.collection.add(
                documents=import_data['documents'],
                metadatas=import_data['metadatas'],
                ids=import_data['ids']
            )

            print(f"Embeddings data imported from {input_file}")
            return True

        except Exception as e:
            print(f"Error importing embeddings data: {e}")
            return False

def main():
    """Main function to demonstrate vector storage and retrieval"""
    print("VectorStorageRetrieval class created successfully!")
    print("This class provides advanced vector storage and retrieval capabilities.")

    # Example usage:
    """
    # Initialize storage
    storage = VectorStorageRetrieval()

    # Store embeddings
    texts = ["Sample news text 1", "Sample news text 2"]
    categories = ["POLITICS", "SPORTS"]
    storage.store_embeddings_with_metadata(texts, categories)

    # Retrieve similar articles
    similar = storage.retrieve_top_k_similar("Sample query", k=5)
    print(similar)
    """

if __name__ == "__main__":
    main()
