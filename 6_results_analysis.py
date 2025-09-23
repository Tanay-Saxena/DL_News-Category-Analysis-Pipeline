"""
Results Collection and Analysis Module
Collects and analyzes results from the news classification pipeline
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import pickle
from collections import defaultdict, Counter

class ResultsAnalyzer:
    def __init__(self,
                 output_dir: str = "results",
                 results_filename: str = "classification_results.csv"):
        """
        Initialize ResultsAnalyzer

        Args:
            output_dir (str): Directory to save results
            results_filename (str): Filename for main results CSV
        """
        self.output_dir = output_dir
        self.results_filename = results_filename
        self.results_df = None
        self.analysis_summary = {}

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

    def collect_prediction_results(self,
                                 input_texts: List[str],
                                 ground_truth_categories: List[str],
                                 predicted_categories: List[str],
                                 deepseek_explanations: List[Dict],
                                 similarity_scores: List[float],
                                 similar_articles: List[List[Dict]],
                                 clustering_labels: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Collect all prediction results into a structured DataFrame

        Args:
            input_texts (List[str]): Input text descriptions
            ground_truth_categories (List[str]): True category labels
            predicted_categories (List[str]): Predicted category labels
            deepseek_explanations (List[Dict]): DeepSeek explanations
            similarity_scores (List[float]): Similarity scores from vector search
            similar_articles (List[List[Dict]]): Similar articles found
            clustering_labels (List[int]): Optional clustering labels

        Returns:
            pd.DataFrame: Structured results DataFrame
        """
        print("Collecting prediction results...")

        # Validate input lengths
        lengths = [len(input_texts), len(ground_truth_categories),
                  len(predicted_categories), len(deepseek_explanations),
                  len(similarity_scores), len(similar_articles)]

        if len(set(lengths)) > 1:
            print(f"Warning: Input lengths don't match: {lengths}")
            min_length = min(lengths)
            input_texts = input_texts[:min_length]
            ground_truth_categories = ground_truth_categories[:min_length]
            predicted_categories = predicted_categories[:min_length]
            deepseek_explanations = deepseek_explanations[:min_length]
            similarity_scores = similarity_scores[:min_length]
            similar_articles = similar_articles[:min_length]
            if clustering_labels:
                clustering_labels = clustering_labels[:min_length]

        # Prepare results data
        results_data = []

        for i in range(len(input_texts)):
            # Extract key information from DeepSeek explanation
            explanation = deepseek_explanations[i] if i < len(deepseek_explanations) else {}

            # Determine match status
            is_match = ground_truth_categories[i] == predicted_categories[i]
            match_status = "Correct" if is_match else "Incorrect"

            # Extract similarity information
            similar_arts = similar_articles[i] if i < len(similar_articles) else []
            similar_categories = [art.get('category', 'Unknown') for art in similar_arts]
            similar_category_counts = Counter(similar_categories)
            most_common_similar_category = similar_category_counts.most_common(1)[0][0] if similar_category_counts else "None"

            # Create result record
            result_record = {
                'index': i,
                'input_text': input_texts[i],
                'ground_truth_category': ground_truth_categories[i],
                'predicted_category': predicted_categories[i],
                'is_match': is_match,
                'match_status': match_status,
                'similarity_score': similarity_scores[i] if i < len(similarity_scores) else 0.0,
                'deepseek_confidence': explanation.get('confidence', 0.0),
                'deepseek_reasoning': explanation.get('reasoning', ''),
                'deepseek_key_indicators': ', '.join(explanation.get('key_indicators', [])),
                'deepseek_alternative_categories': ', '.join(explanation.get('alternative_categories', [])),
                'similar_articles_count': len(similar_arts),
                'most_common_similar_category': most_common_similar_category,
                'similar_category_distribution': dict(similar_category_counts),
                'text_length': len(input_texts[i]),
                'word_count': len(input_texts[i].split()),
                'clustering_label': clustering_labels[i] if clustering_labels and i < len(clustering_labels) else None,
                'timestamp': datetime.now().isoformat()
            }

            # Add detailed explanation fields if available
            if 'explanation' in explanation:
                result_record['detailed_explanation'] = explanation['explanation']
            if 'key_phrases' in explanation:
                result_record['key_phrases'] = ', '.join(explanation['key_phrases'])
            if 'tone_analysis' in explanation:
                result_record['tone_analysis'] = explanation['tone_analysis']
            if 'subject_matter' in explanation:
                result_record['subject_matter'] = explanation['subject_matter']

            results_data.append(result_record)

        # Create DataFrame
        self.results_df = pd.DataFrame(results_data)

        print(f"Collected {len(self.results_df)} prediction results")
        return self.results_df

    def analyze_results(self) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of the results

        Returns:
            Dict: Analysis results
        """
        if self.results_df is None:
            print("Error: No results data available. Call collect_prediction_results() first.")
            return {}

        print("Analyzing results...")

        # Basic accuracy metrics
        total_predictions = len(self.results_df)
        correct_predictions = self.results_df['is_match'].sum()
        accuracy = correct_predictions / total_predictions

        # Category-wise analysis
        category_analysis = self.results_df.groupby('ground_truth_category').agg({
            'is_match': ['count', 'sum', 'mean'],
            'similarity_score': 'mean',
            'deepseek_confidence': 'mean',
            'text_length': 'mean'
        }).round(3)

        category_analysis.columns = ['total_predictions', 'correct_predictions', 'accuracy',
                                   'avg_similarity', 'avg_confidence', 'avg_text_length']

        # Predicted category analysis
        predicted_category_analysis = self.results_df.groupby('predicted_category').agg({
            'is_match': ['count', 'sum', 'mean'],
            'similarity_score': 'mean',
            'deepseek_confidence': 'mean'
        }).round(3)

        predicted_category_analysis.columns = ['total_predictions', 'correct_predictions', 'accuracy',
                                            'avg_similarity', 'avg_confidence']

        # Confusion matrix
        confusion_matrix = pd.crosstab(
            self.results_df['ground_truth_category'],
            self.results_df['predicted_category'],
            margins=True
        )

        # Error analysis
        incorrect_predictions = self.results_df[~self.results_df['is_match']]

        # Most common error patterns
        error_patterns = incorrect_predictions.groupby(['ground_truth_category', 'predicted_category']).size().reset_index(name='count')
        error_patterns = error_patterns.sort_values('count', ascending=False)

        # Similarity score analysis
        similarity_analysis = {
            'mean_similarity': self.results_df['similarity_score'].mean(),
            'std_similarity': self.results_df['similarity_score'].std(),
            'min_similarity': self.results_df['similarity_score'].min(),
            'max_similarity': self.results_df['similarity_score'].max(),
            'similarity_by_match': {
                'correct': self.results_df[self.results_df['is_match']]['similarity_score'].mean(),
                'incorrect': self.results_df[~self.results_df['is_match']]['similarity_score'].mean()
            }
        }

        # DeepSeek confidence analysis
        confidence_analysis = {
            'mean_confidence': self.results_df['deepseek_confidence'].mean(),
            'std_confidence': self.results_df['deepseek_confidence'].std(),
            'confidence_by_match': {
                'correct': self.results_df[self.results_df['is_match']]['deepseek_confidence'].mean(),
                'incorrect': self.results_df[~self.results_df['is_match']]['deepseek_confidence'].mean()
            }
        }

        # Text length analysis
        length_analysis = {
            'mean_length': self.results_df['text_length'].mean(),
            'std_length': self.results_df['text_length'].std(),
            'length_by_match': {
                'correct': self.results_df[self.results_df['is_match']]['text_length'].mean(),
                'incorrect': self.results_df[~self.results_df['is_match']]['text_length'].mean()
            }
        }

        # Clustering analysis (if available)
        clustering_analysis = {}
        if 'clustering_label' in self.results_df.columns and self.results_df['clustering_label'].notna().any():
            cluster_accuracy = self.results_df.groupby('clustering_label')['is_match'].agg(['count', 'sum', 'mean']).round(3)
            cluster_accuracy.columns = ['total_predictions', 'correct_predictions', 'accuracy']
            clustering_analysis['cluster_accuracy'] = cluster_accuracy

        # Compile analysis results
        analysis_results = {
            'overall_metrics': {
                'total_predictions': total_predictions,
                'correct_predictions': correct_predictions,
                'accuracy': accuracy,
                'error_rate': 1 - accuracy
            },
            'category_analysis': category_analysis.to_dict(),
            'predicted_category_analysis': predicted_category_analysis.to_dict(),
            'confusion_matrix': confusion_matrix.to_dict(),
            'error_patterns': error_patterns.to_dict('records'),
            'similarity_analysis': similarity_analysis,
            'confidence_analysis': confidence_analysis,
            'length_analysis': length_analysis,
            'clustering_analysis': clustering_analysis,
            'analysis_timestamp': datetime.now().isoformat()
        }

        self.analysis_summary = analysis_results
        return analysis_results

    def get_example_cases(self,
                         case_type: str = 'all',
                         n_examples: int = 3) -> Dict[str, List[Dict]]:
        """
        Get example cases for different scenarios

        Args:
            case_type (str): Type of cases ('correct', 'incorrect', 'mismatch', 'all')
            n_examples (int): Number of examples to return

        Returns:
            Dict: Example cases organized by type
        """
        if self.results_df is None:
            print("Error: No results data available")
            return {}

        examples = {}

        if case_type in ['correct', 'all']:
            correct_cases = self.results_df[self.results_df['is_match']].head(n_examples)
            examples['correct'] = correct_cases.to_dict('records')

        if case_type in ['incorrect', 'mismatch', 'all']:
            incorrect_cases = self.results_df[~self.results_df['is_match']].head(n_examples)
            examples['incorrect'] = incorrect_cases.to_dict('records')

        # Find specific mismatch cases
        if case_type in ['mismatch', 'all']:
            # Look for cases where predicted category is very different from ground truth
            mismatch_cases = self.results_df[
                (~self.results_df['is_match']) &
                (self.results_df['similarity_score'] < 0.5)
            ].head(n_examples)
            examples['mismatch'] = mismatch_cases.to_dict('records')

        return examples

    def generate_detailed_report(self) -> str:
        """
        Generate a detailed text report of the analysis

        Returns:
            str: Detailed report
        """
        if not self.analysis_summary:
            self.analyze_results()

        report = []
        report.append("=" * 80)
        report.append("NEWS CATEGORY CLASSIFICATION RESULTS REPORT")
        report.append("=" * 80)
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Overall metrics
        overall = self.analysis_summary['overall_metrics']
        report.append("OVERALL PERFORMANCE")
        report.append("-" * 40)
        report.append(f"Total Predictions: {overall['total_predictions']}")
        report.append(f"Correct Predictions: {overall['correct_predictions']}")
        report.append(f"Accuracy: {overall['accuracy']:.3f} ({overall['accuracy']*100:.1f}%)")
        report.append(f"Error Rate: {overall['error_rate']:.3f} ({overall['error_rate']*100:.1f}%)")
        report.append("")

        # Category performance
        report.append("CATEGORY-WISE PERFORMANCE")
        report.append("-" * 40)
        category_analysis = self.analysis_summary['category_analysis']
        for category, metrics in category_analysis.items():
            if isinstance(metrics, dict) and 'accuracy' in metrics:
                report.append(f"{category}: {metrics['accuracy']:.3f} accuracy ({metrics['correct_predictions']}/{metrics['total_predictions']})")
        report.append("")

        # Error patterns
        report.append("TOP ERROR PATTERNS")
        report.append("-" * 40)
        error_patterns = self.analysis_summary['error_patterns'][:10]
        for pattern in error_patterns:
            report.append(f"{pattern['ground_truth_category']} → {pattern['predicted_category']}: {pattern['count']} cases")
        report.append("")

        # Similarity analysis
        sim_analysis = self.analysis_summary['similarity_analysis']
        report.append("SIMILARITY SCORE ANALYSIS")
        report.append("-" * 40)
        report.append(f"Mean Similarity: {sim_analysis['mean_similarity']:.3f}")
        report.append(f"Correct Predictions Mean: {sim_analysis['similarity_by_match']['correct']:.3f}")
        report.append(f"Incorrect Predictions Mean: {sim_analysis['similarity_by_match']['incorrect']:.3f}")
        report.append("")

        # Confidence analysis
        conf_analysis = self.analysis_summary['confidence_analysis']
        report.append("DEEPSEEK CONFIDENCE ANALYSIS")
        report.append("-" * 40)
        report.append(f"Mean Confidence: {conf_analysis['mean_confidence']:.3f}")
        report.append(f"Correct Predictions Mean: {conf_analysis['confidence_by_match']['correct']:.3f}")
        report.append(f"Incorrect Predictions Mean: {conf_analysis['confidence_by_match']['incorrect']:.3f}")
        report.append("")

        return "\n".join(report)

    def save_results(self,
                    save_analysis: bool = True,
                    save_examples: bool = True) -> bool:
        """
        Save results to files

        Args:
            save_analysis (bool): Save analysis summary
            save_examples (bool): Save example cases

        Returns:
            bool: Success status
        """
        try:
            # Save main results DataFrame
            if self.results_df is not None:
                results_path = os.path.join(self.output_dir, self.results_filename)
                self.results_df.to_csv(results_path, index=False)
                print(f"Results saved to {results_path}")

            # Save analysis summary
            if save_analysis and self.analysis_summary:
                analysis_path = os.path.join(self.output_dir, "analysis_summary.json")
                with open(analysis_path, 'w') as f:
                    json.dump(self.analysis_summary, f, indent=2, default=str)
                print(f"Analysis summary saved to {analysis_path}")

                # Save detailed report
                report_path = os.path.join(self.output_dir, "detailed_report.txt")
                with open(report_path, 'w') as f:
                    f.write(self.generate_detailed_report())
                print(f"Detailed report saved to {report_path}")

            # Save example cases
            if save_examples:
                examples = self.get_example_cases('all', 5)
                examples_path = os.path.join(self.output_dir, "example_cases.json")
                with open(examples_path, 'w') as f:
                    json.dump(examples, f, indent=2, default=str)
                print(f"Example cases saved to {examples_path}")

            return True

        except Exception as e:
            print(f"Error saving results: {e}")
            return False

    def load_results(self,
                    results_file: Optional[str] = None) -> bool:
        """
        Load results from file

        Args:
            results_file (str): Path to results file

        Returns:
            bool: Success status
        """
        try:
            if results_file is None:
                results_file = os.path.join(self.output_dir, self.results_filename)

            self.results_df = pd.read_csv(results_file)
            print(f"Results loaded from {results_file}")

            # Try to load analysis summary
            analysis_file = os.path.join(self.output_dir, "analysis_summary.json")
            if os.path.exists(analysis_file):
                with open(analysis_file, 'r') as f:
                    self.analysis_summary = json.load(f)
                print("Analysis summary loaded")

            return True

        except Exception as e:
            print(f"Error loading results: {e}")
            return False

def main():
    """Main function to demonstrate results analysis"""
    print("ResultsAnalyzer class created successfully!")
    print("This class provides comprehensive analysis of classification results.")

    # Example usage:
    """
    # Initialize analyzer
    analyzer = ResultsAnalyzer()

    # Collect results (assuming you have prediction data)
    # results_df = analyzer.collect_prediction_results(
    #     input_texts, ground_truth, predictions, explanations,
    #     similarities, similar_articles
    # )

    # Analyze results
    # analysis = analyzer.analyze_results()

    # Save results
    # analyzer.save_results()
    """

if __name__ == "__main__":
    main()
