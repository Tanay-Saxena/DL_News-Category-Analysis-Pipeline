"""
Embedding Generation Module using ChromaDB and LangChain
Creates vector embeddings for news descriptions and stores them in ChromaDB
"""

import chromadb
from chromadb.config import Settings
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import os
import json
from tqdm import tqdm
import pickle

class EmbeddingGenerator:
    def __init__(self,
                 collection_name: str = "news_embeddings",
                 persist_directory: str = "./chroma_db",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize EmbeddingGenerator

        Args:
            collection_name (str): Name of the ChromaDB collection
            persist_directory (str): Directory to persist ChromaDB data
            embedding_model (str): Name of the embedding model to use
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        # Initialize collection
        self.collection = None
        self._initialize_collection()

        # Store metadata for analysis
        self.metadata = {
            'total_documents': 0,
            'categories': set(),
            'embedding_dimension': None,
            'model_name': embedding_model
        }

    def _initialize_collection(self):
        """Initialize or get existing ChromaDB collection"""
        try:
            # Try to get existing collection
            self.collection = self.client.get_collection(name=self.collection_name)
            print(f"Loaded existing collection: {self.collection_name}")

            # Update metadata
            count = self.collection.count()
            self.metadata['total_documents'] = count
            print(f"Collection contains {count} documents")

        except Exception:
            # Create new collection
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "News category embeddings with metadata"}
            )
            print(f"Created new collection: {self.collection_name}")

    def generate_embeddings_from_dataframe(self,
                                         df: pd.DataFrame,
                                         text_column: str = 'short_description',
                                         category_column: str = 'category',
                                         batch_size: int = 100) -> bool:
        """
        Generate embeddings for all texts in a DataFrame

        Args:
            df (pd.DataFrame): DataFrame containing text and category data
            text_column (str): Name of the column containing text
            category_column (str): Name of the column containing categories
            batch_size (int): Number of documents to process in each batch

        Returns:
            bool: True if successful, False otherwise
        """
        if df is None or df.empty:
            print("Error: DataFrame is empty or None")
            return False

        print(f"Generating embeddings for {len(df)} documents...")
        print(f"Using model: {self.embedding_model}")

        # Prepare data for batch processing
        texts = df[text_column].tolist()
        categories = df[category_column].tolist()
        ids = [f"doc_{i}" for i in range(len(texts))]

        # Process in batches
        total_batches = (len(texts) + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(total_batches), desc="Processing batches"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(texts))

            batch_texts = texts[start_idx:end_idx]
            batch_categories = categories[start_idx:end_idx]
            batch_ids = ids[start_idx:end_idx]

            # Create metadata for this batch
            batch_metadata = [
                {
                    "category": cat,
                    "text_length": len(text),
                    "batch_id": batch_idx
                }
                for cat, text in zip(batch_categories, batch_texts)
            ]

            try:
                # Add documents to collection
                self.collection.add(
                    documents=batch_texts,
                    metadatas=batch_metadata,
                    ids=batch_ids
                )

                # Update metadata
                self.metadata['categories'].update(batch_categories)

            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue

        # Update final metadata
        self.metadata['total_documents'] = self.collection.count()
        self.metadata['categories'] = list(self.metadata['categories'])

        print(f"Successfully generated embeddings for {self.metadata['total_documents']} documents")
        print(f"Categories found: {len(self.metadata['categories'])}")

        return True

    def get_similar_documents(self,
                            query_text: str,
                            n_results: int = 5,
                            where_clause: Optional[Dict] = None) -> List[Dict]:
        """
        Retrieve similar documents based on query text

        Args:
            query_text (str): Text to search for
            n_results (int): Number of similar documents to return
            where_clause (Dict): Optional metadata filter

        Returns:
            List[Dict]: List of similar documents with metadata
        """
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=where_clause
            )

            # Format results
            similar_docs = []
            for i in range(len(results['documents'][0])):
                doc = {
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'distance': results['distances'][0][i],
                    'metadata': results['metadatas'][0][i]
                }
                similar_docs.append(doc)

            return similar_docs

        except Exception as e:
            print(f"Error retrieving similar documents: {e}")
            return []

    def get_embeddings_by_category(self, category: str) -> List[Dict]:
        """
        Get all embeddings for a specific category

        Args:
            category (str): Category to filter by

        Returns:
            List[Dict]: List of documents in the category
        """
        try:
            results = self.collection.get(
                where={"category": category}
            )

            docs = []
            for i in range(len(results['documents'])):
                doc = {
                    'id': results['ids'][i],
                    'text': results['documents'][i],
                    'metadata': results['metadatas'][i]
                }
                docs.append(doc)

            return docs

        except Exception as e:
            print(f"Error retrieving documents for category {category}: {e}")
            return []

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection

        Returns:
            Dict: Collection statistics
        """
        try:
            count = self.collection.count()

            # Get category distribution
            all_docs = self.collection.get()
            categories = [doc['category'] for doc in all_docs['metadatas']]
            category_counts = pd.Series(categories).value_counts().to_dict()

            stats = {
                'total_documents': count,
                'unique_categories': len(set(categories)),
                'category_distribution': category_counts,
                'embedding_model': self.embedding_model,
                'collection_name': self.collection_name
            }

            return stats

        except Exception as e:
            print(f"Error getting collection stats: {e}")
            return {}

    def save_metadata(self, filepath: str = "embedding_metadata.json"):
        """Save metadata to file"""
        try:
            # Convert set to list for JSON serialization
            metadata_copy = self.metadata.copy()
            if 'categories' in metadata_copy and isinstance(metadata_copy['categories'], set):
                metadata_copy['categories'] = list(metadata_copy['categories'])

            with open(filepath, 'w') as f:
                json.dump(metadata_copy, f, indent=2)

            print(f"Metadata saved to {filepath}")
            return True

        except Exception as e:
            print(f"Error saving metadata: {e}")
            return False

    def load_metadata(self, filepath: str = "embedding_metadata.json"):
        """Load metadata from file"""
        try:
            with open(filepath, 'r') as f:
                metadata = json.load(f)

            # Convert list back to set
            if 'categories' in metadata and isinstance(metadata['categories'], list):
                metadata['categories'] = set(metadata['categories'])

            self.metadata.update(metadata)
            print(f"Metadata loaded from {filepath}")
            return True

        except Exception as e:
            print(f"Error loading metadata: {e}")
            return False

    def clear_collection(self):
        """Clear all documents from the collection"""
        try:
            # Delete the collection
            self.client.delete_collection(name=self.collection_name)

            # Recreate empty collection
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "News category embeddings with metadata"}
            )

            # Reset metadata
            self.metadata = {
                'total_documents': 0,
                'categories': set(),
                'embedding_dimension': None,
                'model_name': self.embedding_model
            }

            print("Collection cleared successfully")
            return True

        except Exception as e:
            print(f"Error clearing collection: {e}")
            return False

def main():
    """Main function to demonstrate embedding generation"""
    # This would typically be called after data preparation
    print("EmbeddingGenerator class created successfully!")
    print("Use this class to generate and store embeddings for your news data.")

    # Example usage (commented out since we need data first):
    """
    # Initialize generator
    generator = EmbeddingGenerator()

    # Load your processed data
    train_df = pd.read_csv('processed_data/train_data.csv')

    # Generate embeddings
    generator.generate_embeddings_from_dataframe(train_df)

    # Get stats
    stats = generator.get_collection_stats()
    print(stats)
    """

if __name__ == "__main__":
    main()
