"""
Data Preparation Module for News Category Classification
Loads Kaggle News Category Dataset, performs cleaning, and splits into train/test
"""

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
import os

class DataPreparator:
    def __init__(self, data_path=None):
        """
        Initialize DataPreparator

        Args:
            data_path (str): Path to the CSV file. If None, will look for 'News_Category_Dataset_v3.csv'
        """
        self.data_path = data_path or 'News_Category_Dataset_v3.csv'
        self.df = None
        self.train_df = None
        self.test_df = None

    def load_data(self):
        """Load the Kaggle News Category Dataset"""
        try:
            print(f"Loading data from {self.data_path}...")
            self.df = pd.read_csv(self.data_path)
            print(f"Loaded {len(self.df)} rows and {len(self.df.columns)} columns")
            print(f"Columns: {list(self.df.columns)}")
            return True
        except FileNotFoundError:
            print(f"Error: Could not find {self.data_path}")
            print("Please ensure the Kaggle News Category Dataset is in the project directory")
            return False
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def clean_data(self):
        """
        Clean the dataset by:
        1. Keeping only 'short_description' and 'category' columns
        2. Removing null values
        3. Converting to lowercase
        4. Stripping special characters
        """
        if self.df is None:
            print("Error: No data loaded. Call load_data() first.")
            return False

        print("Cleaning data...")

        # Keep only required columns
        required_columns = ['short_description', 'category']
        missing_columns = [col for col in required_columns if col not in self.df.columns]

        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            print(f"Available columns: {list(self.df.columns)}")
            return False

        # Select only required columns
        self.df = self.df[required_columns].copy()
        print(f"Selected columns: {list(self.df.columns)}")

        # Remove null values
        initial_count = len(self.df)
        self.df = self.df.dropna()
        print(f"Removed {initial_count - len(self.df)} rows with null values")

        # Clean text data
        def clean_text(text):
            if pd.isna(text):
                return ""
            # Convert to lowercase
            text = str(text).lower()
            # Remove special characters but keep spaces and basic punctuation
            text = re.sub(r'[^a-zA-Z0-9\s.,!?-]', '', text)
            # Remove extra whitespace
            text = ' '.join(text.split())
            return text

        # Apply cleaning to short_description
        self.df['short_description'] = self.df['short_description'].apply(clean_text)

        # Remove empty descriptions after cleaning
        self.df = self.df[self.df['short_description'].str.len() > 0]

        print(f"After cleaning: {len(self.df)} rows")
        print(f"Category distribution:")
        print(self.df['category'].value_counts().head(10))

        return True

    def split_data(self, test_size=0.2, random_state=42):
        """
        Split data into train and test sets

        Args:
            test_size (float): Proportion of data for test set
            random_state (int): Random seed for reproducibility
        """
        if self.df is None:
            print("Error: No data loaded. Call load_data() and clean_data() first.")
            return False

        print(f"Splitting data into train/test ({1-test_size:.0%}/{test_size:.0%})...")

        # Stratified split to maintain category distribution
        self.train_df, self.test_df = train_test_split(
            self.df,
            test_size=test_size,
            random_state=random_state,
            stratify=self.df['category']
        )

        print(f"Train set: {len(self.train_df)} rows")
        print(f"Test set: {len(self.test_df)} rows")

        # Verify category distribution is maintained
        train_cats = self.train_df['category'].value_counts(normalize=True).sort_index()
        test_cats = self.test_df['category'].value_counts(normalize=True).sort_index()

        print("\nCategory distribution comparison:")
        print("Train vs Test (top 10 categories):")
        comparison_df = pd.DataFrame({
            'Train': train_cats,
            'Test': test_cats
        }).head(10)
        print(comparison_df)

        return True

    def get_data_summary(self):
        """Get summary statistics of the processed data"""
        if self.df is None:
            return None

        summary = {
            'total_rows': len(self.df),
            'unique_categories': self.df['category'].nunique(),
            'avg_description_length': self.df['short_description'].str.len().mean(),
            'min_description_length': self.df['short_description'].str.len().min(),
            'max_description_length': self.df['short_description'].str.len().max(),
            'category_distribution': self.df['category'].value_counts().to_dict()
        }

        return summary

    def save_processed_data(self, output_dir='processed_data'):
        """Save processed train and test data"""
        if self.train_df is None or self.test_df is None:
            print("Error: No processed data to save. Run split_data() first.")
            return False

        os.makedirs(output_dir, exist_ok=True)

        train_path = os.path.join(output_dir, 'train_data.csv')
        test_path = os.path.join(output_dir, 'test_data.csv')

        self.train_df.to_csv(train_path, index=False)
        self.test_df.to_csv(test_path, index=False)

        print(f"Saved train data to {train_path}")
        print(f"Saved test data to {test_path}")

        return True

def main():
    """Main function to demonstrate data preparation"""
    # Initialize data preparator
    preparator = DataPreparator()

    # Load data
    if not preparator.load_data():
        return

    # Clean data
    if not preparator.clean_data():
        return

    # Split data
    if not preparator.split_data():
        return

    # Get summary
    summary = preparator.get_data_summary()
    print("\nData Summary:")
    print(f"Total rows: {summary['total_rows']}")
    print(f"Unique categories: {summary['unique_categories']}")
    print(f"Average description length: {summary['avg_description_length']:.1f} characters")

    # Save processed data
    preparator.save_processed_data()

    return preparator

if __name__ == "__main__":
    preparator = main()
