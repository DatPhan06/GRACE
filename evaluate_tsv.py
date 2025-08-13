#!/usr/bin/env python3
"""
Comprehensive evaluation script for REDIAL recommendation system results.
Analyzes output/REDIAL/test_reranking_no_plot/gemini-2.0-flash_recall@10_500sample.tsv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class REDIALEvaluator:
    def __init__(self, tsv_file_path):
        """
        Initialize the evaluator with the TSV file path.
        
        Args:
            tsv_file_path (str): Path to the TSV file to analyze
        """
        self.tsv_file_path = Path(tsv_file_path)
        self.df = None
        self.load_data()
    
    def load_data(self):
        """Load and preprocess the TSV data."""
        try:
            print(f"Loading data from {self.tsv_file_path}...")
            self.df = pd.read_csv(self.tsv_file_path, sep='\t', encoding='utf-8')
            print(f"Successfully loaded {len(self.df)} records")
            
            # Basic info about the dataset
            print(f"Columns: {list(self.df.columns)}")
            print(f"Data shape: {self.df.shape}")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def basic_statistics(self):
        """Generate basic statistics about the dataset."""
        print("\n" + "="*50)
        print("BASIC STATISTICS")
        print("="*50)
        
        # Dataset overview
        print(f"Total number of recommendations: {len(self.df)}")
        print(f"Number of unique items recommended: {self.df['recommend_item'].nunique()}")
        
        # Recall statistics
        print(f"\nRecall Statistics:")
        print(f"Mean recall: {self.df['recall'].mean():.4f}")
        print(f"Median recall: {self.df['recall'].median():.4f}")
        print(f"Standard deviation: {self.df['recall'].std():.4f}")
        print(f"Min recall: {self.df['recall'].min():.4f}")
        print(f"Max recall: {self.df['recall'].max():.4f}")
        
        # Distribution of recall values
        recall_counts = self.df['recall'].value_counts().sort_index()
        print(f"\nRecall Value Distribution:")
        for recall_val, count in recall_counts.items():
            percentage = (count / len(self.df)) * 100
            print(f"  Recall {recall_val}: {count} samples ({percentage:.1f}%)")
        
        return {
            'total_recommendations': len(self.df),
            'unique_items': self.df['recommend_item'].nunique(),
            'mean_recall': self.df['recall'].mean(),
            'median_recall': self.df['recall'].median(),
            'std_recall': self.df['recall'].std(),
            'recall_distribution': recall_counts.to_dict()
        }
    
    def analyze_recommendations(self):
        """Analyze the recommended items and their patterns."""
        print("\n" + "="*50)
        print("RECOMMENDATION ANALYSIS")
        print("="*50)
        
        # Most frequently recommended items
        print("Top 10 Most Recommended Items:")
        top_items = self.df['recommend_item'].value_counts().head(10)
        for i, (item, count) in enumerate(top_items.items(), 1):
            print(f"  {i:2d}. {item}: {count} times")
        
        # Analyze recommendation success by item
        print(f"\nRecommendation Success Analysis:")
        item_performance = self.df.groupby('recommend_item').agg({
            'recall': ['count', 'mean', 'sum']
        }).round(4)
        item_performance.columns = ['frequency', 'avg_recall', 'total_recall']
        item_performance = item_performance.sort_values('frequency', ascending=False)
        
        print("Top 10 Items by Performance:")
        for item in item_performance.head(10).index:
            freq = item_performance.loc[item, 'frequency']
            avg_recall = item_performance.loc[item, 'avg_recall']
            print(f"  {item}: {freq} recs, avg recall: {avg_recall:.3f}")
        
        return item_performance
    
    def analyze_movie_lists(self):
        """Analyze the recommend_movie_list and movie_candidate_list columns."""
        print("\n" + "="*50)
        print("MOVIE LIST ANALYSIS")
        print("="*50)
        
        # Analyze recommended movie list lengths
        self.df['rec_list_length'] = self.df['recommend_movie_list'].str.split('|').str.len()
        self.df['candidate_list_length'] = self.df['movie_candidate_list'].str.split('|').str.len()
        
        print("Recommended Movie List Statistics:")
        print(f"  Mean length: {self.df['rec_list_length'].mean():.2f}")
        print(f"  Median length: {self.df['rec_list_length'].median():.0f}")
        print(f"  Min length: {self.df['rec_list_length'].min()}")
        print(f"  Max length: {self.df['rec_list_length'].max()}")
        
        print("\nCandidate Movie List Statistics:")
        print(f"  Mean length: {self.df['candidate_list_length'].mean():.2f}")
        print(f"  Median length: {self.df['candidate_list_length'].median():.0f}")
        print(f"  Min length: {self.df['candidate_list_length'].min()}")
        print(f"  Max length: {self.df['candidate_list_length'].max()}")
        
        # Analyze relationship between list lengths and recall
        print(f"\nCorrelation between list lengths and recall:")
        rec_corr = self.df['rec_list_length'].corr(self.df['recall'])
        cand_corr = self.df['candidate_list_length'].corr(self.df['recall'])
        print(f"  Recommendation list length vs recall: {rec_corr:.4f}")
        print(f"  Candidate list length vs recall: {cand_corr:.4f}")
        
        return {
            'rec_list_stats': self.df['rec_list_length'].describe(),
            'candidate_list_stats': self.df['candidate_list_length'].describe(),
            'correlations': {'rec_length_recall': rec_corr, 'cand_length_recall': cand_corr}
        }
    
    def analyze_conversations(self):
        """Analyze the summarized conversation patterns."""
        print("\n" + "="*50)
        print("CONVERSATION ANALYSIS")
        print("="*50)
        
        # Conversation length analysis
        self.df['conversation_length'] = self.df['summarized_conversation'].str.len()
        
        print("Conversation Length Statistics:")
        print(f"  Mean length: {self.df['conversation_length'].mean():.0f} characters")
        print(f"  Median length: {self.df['conversation_length'].median():.0f} characters")
        print(f"  Min length: {self.df['conversation_length'].min()} characters")
        print(f"  Max length: {self.df['conversation_length'].max()} characters")
        
        # Common patterns in conversations
        print(f"\nCommon conversation patterns:")
        
        # Extract genres mentioned
        genre_patterns = [
            'comedy', 'horror', 'action', 'drama', 'romance', 'thriller', 
            'sci-fi', 'fantasy', 'animation', 'documentary'
        ]
        
        for genre in genre_patterns:
            count = self.df['summarized_conversation'].str.contains(genre, case=False).sum()
            if count > 0:
                print(f"  Mentions '{genre}': {count} conversations")
        
        # Preference indicators
        preference_patterns = [
            'similar to', 'like', 'enjoy', 'preference', 'looking for'
        ]
        
        print(f"\nPreference indicators:")
        for pattern in preference_patterns:
            count = self.df['summarized_conversation'].str.contains(pattern, case=False).sum()
            if count > 0:
                print(f"  Contains '{pattern}': {count} conversations")
        
        return {
            'conversation_length_stats': self.df['conversation_length'].describe(),
            'genre_mentions': {genre: self.df['summarized_conversation'].str.contains(genre, case=False).sum() 
                              for genre in genre_patterns},
            'preference_patterns': {pattern: self.df['summarized_conversation'].str.contains(pattern, case=False).sum() 
                                   for pattern in preference_patterns}
        }
    
    def recall_performance_analysis(self):
        """Detailed analysis of recall performance."""
        print("\n" + "="*50)
        print("RECALL PERFORMANCE ANALYSIS")
        print("="*50)
        
        # Perfect recall analysis
        perfect_recall = self.df[self.df['recall'] == 1.0]
        print(f"Perfect Recall (1.0) Cases: {len(perfect_recall)} ({len(perfect_recall)/len(self.df)*100:.1f}%)")
        
        if len(perfect_recall) > 0:
            print(f"Top items with perfect recall:")
            perfect_items = perfect_recall['recommend_item'].value_counts().head(5)
            for item, count in perfect_items.items():
                print(f"  {item}: {count} times")
        
        # Zero recall analysis
        zero_recall = self.df[self.df['recall'] == 0.0]
        print(f"\nZero Recall (0.0) Cases: {len(zero_recall)} ({len(zero_recall)/len(self.df)*100:.1f}%)")
        
        if len(zero_recall) > 0:
            print(f"Top items with zero recall:")
            zero_items = zero_recall['recommend_item'].value_counts().head(5)
            for item, count in zero_items.items():
                print(f"  {item}: {count} times")
        
        # Recall distribution by bins
        bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        self.df['recall_bin'] = pd.cut(self.df['recall'], bins=bins, include_lowest=True)
        recall_dist = self.df['recall_bin'].value_counts().sort_index()
        
        print(f"\nRecall Distribution by Bins:")
        for bin_range, count in recall_dist.items():
            percentage = (count / len(self.df)) * 100
            print(f"  {bin_range}: {count} samples ({percentage:.1f}%)")
        
        return {
            'perfect_recall_count': len(perfect_recall),
            'zero_recall_count': len(zero_recall),
            'recall_distribution': recall_dist.to_dict()
        }
    
    def create_visualizations(self, output_dir="plots"):
        """Create visualizations for the analysis."""
        print(f"\n" + "="*50)
        print("CREATING VISUALIZATIONS")
        print("="*50)
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Recall distribution histogram
        plt.figure(figsize=(10, 6))
        plt.hist(self.df['recall'], bins=20, alpha=0.7, edgecolor='black')
        plt.title('Distribution of Recall Values')
        plt.xlabel('Recall')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.savefig(output_path / 'recall_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Top recommended items
        plt.figure(figsize=(12, 8))
        top_items = self.df['recommend_item'].value_counts().head(15)
        top_items.plot(kind='barh')
        plt.title('Top 15 Most Recommended Items')
        plt.xlabel('Frequency')
        plt.tight_layout()
        plt.savefig(output_path / 'top_recommended_items.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Recall by recommendation frequency
        plt.figure(figsize=(10, 6))
        item_stats = self.df.groupby('recommend_item').agg({
            'recall': ['count', 'mean']
        })
        item_stats.columns = ['frequency', 'avg_recall']
        
        # Filter items with at least 2 recommendations for cleaner plot
        item_stats_filtered = item_stats[item_stats['frequency'] >= 2]
        
        plt.scatter(item_stats_filtered['frequency'], item_stats_filtered['avg_recall'], alpha=0.6)
        plt.xlabel('Recommendation Frequency')
        plt.ylabel('Average Recall')
        plt.title('Average Recall vs Recommendation Frequency')
        plt.grid(True, alpha=0.3)
        plt.savefig(output_path / 'recall_vs_frequency.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. List length analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Rec list length distribution
        axes[0, 0].hist(self.df['rec_list_length'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Recommendation List Length Distribution')
        axes[0, 0].set_xlabel('List Length')
        axes[0, 0].set_ylabel('Frequency')
        
        # Candidate list length distribution
        axes[0, 1].hist(self.df['candidate_list_length'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Candidate List Length Distribution')
        axes[0, 1].set_xlabel('List Length')
        axes[0, 1].set_ylabel('Frequency')
        
        # Rec list length vs recall
        axes[1, 0].scatter(self.df['rec_list_length'], self.df['recall'], alpha=0.5)
        axes[1, 0].set_xlabel('Recommendation List Length')
        axes[1, 0].set_ylabel('Recall')
        axes[1, 0].set_title('Recall vs Recommendation List Length')
        
        # Candidate list length vs recall
        axes[1, 1].scatter(self.df['candidate_list_length'], self.df['recall'], alpha=0.5)
        axes[1, 1].set_xlabel('Candidate List Length')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].set_title('Recall vs Candidate List Length')
        
        plt.tight_layout()
        plt.savefig(output_path / 'list_length_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Recall performance heatmap
        if len(self.df['recall'].unique()) > 1:
            plt.figure(figsize=(10, 8))
            
            # Create a correlation matrix of numerical features
            numeric_features = ['recall', 'rec_list_length', 'candidate_list_length', 'conversation_length']
            corr_matrix = self.df[numeric_features].corr()
            
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, fmt='.3f')
            plt.title('Correlation Matrix of Features')
            plt.tight_layout()
            plt.savefig(output_path / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Visualizations saved in '{output_path}' directory")
        
        return output_path
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report."""
        print("\n" + "="*70)
        print("SUMMARY REPORT")
        print("="*70)
        
        total_samples = len(self.df)
        mean_recall = self.df['recall'].mean()
        median_recall = self.df['recall'].median()
        perfect_recall_pct = (self.df['recall'] == 1.0).sum() / total_samples * 100
        zero_recall_pct = (self.df['recall'] == 0.0).sum() / total_samples * 100
        
        print(f"Dataset Overview:")
        print(f"  • Total samples: {total_samples:,}")
        print(f"  • Unique recommended items: {self.df['recommend_item'].nunique():,}")
        print(f"  • File: {self.tsv_file_path.name}")
        
        print(f"\nPerformance Metrics:")
        print(f"  • Mean Recall: {mean_recall:.4f}")
        print(f"  • Median Recall: {median_recall:.4f}")
        print(f"  • Perfect Recall (1.0): {perfect_recall_pct:.1f}% of samples")
        print(f"  • Zero Recall (0.0): {zero_recall_pct:.1f}% of samples")
        
        print(f"\nData Characteristics:")
        print(f"  • Avg recommendation list length: {self.df['rec_list_length'].mean():.1f}")
        print(f"  • Avg candidate list length: {self.df['candidate_list_length'].mean():.1f}")
        print(f"  • Avg conversation length: {self.df['conversation_length'].mean():.0f} chars")
        
        # Top performing item
        item_performance = self.df.groupby('recommend_item').agg({
            'recall': ['count', 'mean']
        })
        item_performance.columns = ['frequency', 'avg_recall']
        item_performance = item_performance[item_performance['frequency'] >= 2]  # At least 2 recommendations
        
        if not item_performance.empty:
            best_item = item_performance.loc[item_performance['avg_recall'].idxmax()]
            print(f"\nBest Performing Item (min 2 recs):")
            print(f"  • Item: {item_performance['avg_recall'].idxmax()}")
            print(f"  • Average recall: {best_item['avg_recall']:.4f}")
            print(f"  • Frequency: {best_item['frequency']:.0f}")
        
        # System assessment
        print(f"\nSystem Assessment:")
        if mean_recall >= 0.8:
            assessment = "Excellent"
        elif mean_recall >= 0.6:
            assessment = "Good"
        elif mean_recall >= 0.4:
            assessment = "Fair"
        else:
            assessment = "Needs Improvement"
        
        print(f"  • Overall Performance: {assessment}")
        print(f"  • Success Rate: {perfect_recall_pct:.1f}% perfect matches")
        print(f"  • Failure Rate: {zero_recall_pct:.1f}% zero matches")
        
        return {
            'total_samples': total_samples,
            'mean_recall': mean_recall,
            'median_recall': median_recall,
            'perfect_recall_pct': perfect_recall_pct,
            'zero_recall_pct': zero_recall_pct,
            'assessment': assessment
        }
    
    def run_full_analysis(self, create_plots=True):
        """Run the complete analysis pipeline."""
        print("Starting comprehensive analysis of REDIAL recommendation results...")
        
        # Run all analyses
        basic_stats = self.basic_statistics()
        item_analysis = self.analyze_recommendations()
        list_analysis = self.analyze_movie_lists()
        conv_analysis = self.analyze_conversations()
        recall_analysis = self.recall_performance_analysis()
        
        if create_plots:
            plot_dir = self.create_visualizations()
        
        summary = self.generate_summary_report()
        
        print(f"\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        
        return {
            'basic_stats': basic_stats,
            'item_analysis': item_analysis,
            'list_analysis': list_analysis,
            'conversation_analysis': conv_analysis,
            'recall_analysis': recall_analysis,
            'summary': summary
        }


def main():
    """Main function to run the evaluation."""
    tsv_file = "output/REDIAL/test/gemini-2.0-flash_recall@10_500sample.tsv"
    
    try:
        evaluator = REDIALEvaluator(tsv_file)
        results = evaluator.run_full_analysis(create_plots=True)
        
        print(f"\nAnalysis complete! Check the 'plots' directory for visualizations.")
        
    except FileNotFoundError:
        print(f"Error: Could not find the file {tsv_file}")
        print("Please ensure the file exists and the path is correct.")
    except Exception as e:
        print(f"An error occurred during analysis: {e}")
        raise


if __name__ == "__main__":
    main() 