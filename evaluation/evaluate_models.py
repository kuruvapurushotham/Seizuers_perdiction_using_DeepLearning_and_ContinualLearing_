"""
Comprehensive Model Evaluation for Seizure Prediction System
Supports single model evaluation, cross-validation, and continual learning assessment
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import StratifiedKFold
import time

# Add parent to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.network import create_model
from models.ewc import EWCTrainer
from training.train import EEGDataset, collate_batch


class ModelEvaluator:
    """Comprehensive model evaluation class"""
    
    def __init__(self, device: str = 'auto'):
        """
        Initialize evaluator
        
        Args:
            device: Device to use ('auto', 'cpu', 'cuda')
        """
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        self.results = {}
        
    def load_model(self, model_path: str, model_name: str = None, in_channels: int = 19, num_classes: int = 2):
        """
        Load trained model from checkpoint
        
        Args:
            model_path: Path to model checkpoint
            model_name: Model architecture name (if not in checkpoint)
            in_channels: Number of input channels
            num_classes: Number of output classes
            
        Returns:
            Loaded model
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get model info from checkpoint if available
        if 'model_kwargs' in checkpoint:
            model_kwargs = checkpoint['model_kwargs']
            model_name = model_kwargs.get('model_name', 'eegnet')
            in_channels = model_kwargs.get('in_channels', in_channels)
            num_classes = model_kwargs.get('num_classes', num_classes)
        
        # Create model
        model = create_model(model_name, in_channels=in_channels, num_classes=num_classes)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(self.device)
        model.eval()
        
        print(f"Loaded {model_name} model from {model_path}")
        return model
    
    def evaluate_single_model(self, model: nn.Module, dataloader: DataLoader, 
                           model_name: str = "model") -> Dict[str, Any]:
        """
        Evaluate a single model on test data
        
        Args:
            model: Trained model
            dataloader: Test data loader
            model_name: Name for reporting
            
        Returns:
            Dictionary of evaluation metrics
        """
        model.eval()
        all_predictions = []
        all_probabilities = []
        all_labels = []
        inference_times = []
        
        print(f"Evaluating {model_name}...")
        
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(dataloader):
                data, targets = data.to(self.device), targets.to(self.device)
                
                # Measure inference time
                start_time = time.time()
                outputs = model(data)
                inference_time = time.time() - start_time
                
                # Get predictions and probabilities
                probabilities = torch.softmax(outputs, dim=1)
                predictions = outputs.argmax(dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_labels.extend(targets.cpu().numpy())
                inference_times.append(inference_time)
                
                if (batch_idx + 1) % 50 == 0:
                    print(f"Processed {batch_idx + 1} batches...")
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        all_labels = np.array(all_labels)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        # For binary classification, calculate additional metrics
        if len(np.unique(all_labels)) == 2:
            auc_roc = roc_auc_score(all_labels, all_probabilities[:, 1])
            sensitivity = recall_score(all_labels, all_predictions)  # True positive rate
            specificity = recall_score(all_labels, all_predictions, pos_label=0)  # True negative rate
        else:
            auc_roc = roc_auc_score(all_labels, all_probabilities, multi_class='ovr')
            sensitivity = recall
            specificity = recall
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        # Inference statistics
        avg_inference_time = np.mean(inference_times)
        std_inference_time = np.std(inference_times)
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'confusion_matrix': cm.tolist(),
            'avg_inference_time': avg_inference_time,
            'std_inference_time': std_inference_time,
            'total_samples': len(all_labels),
            'predictions': all_predictions.tolist(),
            'probabilities': all_probabilities.tolist(),
            'labels': all_labels.tolist()
        }
        
        self.results[model_name] = results
        return results
    
    def cross_validate_model(self, model_name: str, data_dir: str, labels_csv: str,
                           n_folds: int = 5, **model_kwargs) -> Dict[str, Any]:
        """
        Perform cross-validation on a model architecture
        
        Args:
            model_name: Model architecture name
            data_dir: Directory containing data files
            labels_csv: CSV file with labels
            n_folds: Number of cross-validation folds
            **model_kwargs: Additional model parameters
            
        Returns:
            Cross-validation results
        """
        print(f"Performing {n_fold}s-fold cross-validation for {model_name}...")
        
        # Load dataset
        dataset = EEGDataset(Path(data_dir), Path(labels_csv))
        
        # Extract labels for stratified splitting
        labels = [item[1] for item in dataset.items]
        
        # Initialize cross-validation
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        fold_results = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(range(len(dataset)), labels)):
            print(f"\nFold {fold_idx + 1}/{n_folds}")
            
            # Create train/val subsets
            train_items = [dataset.items[i] for i in train_idx]
            val_items = [dataset.items[i] for i in val_idx]
            
            # Create datasets
            train_dataset = EEGDataset.__new__(EEGDataset)
            train_dataset.items = train_items
            
            val_dataset = EEGDataset.__new__(EEGDataset)
            val_dataset.items = val_items
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_batch)
            val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_batch)
            
            # Get input dimensions from a sample
            sample_data, _ = next(iter(train_loader))
            in_channels = sample_data.shape[1]
            
            # Create and train model
            model = create_model(model_name, in_channels=in_channels, num_classes=2, **model_kwargs)
            model.to(self.device)
            
            # Simple training for this fold
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()
            
            # Train for a few epochs
            model.train()
            for epoch in range(10):  # Quick training for CV
                for data, targets in train_loader:
                    data, targets = data.to(self.device), targets.to(self.device)
                    optimizer.zero_grad()
                    outputs = model(data)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
            
            # Evaluate on validation set
            fold_result = self.evaluate_single_model(model, val_loader, f"{model_name}_fold_{fold_idx+1}")
            fold_results.append(fold_result)
        
        # Aggregate results
        cv_results = self._aggregate_cv_results(fold_results, model_name)
        
        return cv_results
    
    def _aggregate_cv_results(self, fold_results: List[Dict], model_name: str) -> Dict[str, Any]:
        """Aggregate cross-validation results"""
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc', 'sensitivity', 'specificity']
        
        aggregated = {
            'model_name': model_name,
            'n_folds': len(fold_results),
            'fold_results': fold_results
        }
        
        for metric in metrics:
            values = [fold[metric] for fold in fold_results]
            aggregated[f'{metric}_mean'] = np.mean(values)
            aggregated[f'{metric}_std'] = np.std(values)
            aggregated[f'{metric}_values'] = values
        
        return aggregated
    
    def compare_models(self, model_paths: List[str], test_dataloader: DataLoader, 
                      model_names: List[str] = None) -> Dict[str, Any]:
        """
        Compare multiple trained models
        
        Args:
            model_paths: List of model checkpoint paths
            test_dataloader: Test data loader
            model_names: List of model names (optional)
            
        Returns:
            Comparison results
        """
        if model_names is None:
            model_names = [f"model_{i+1}" for i in range(len(model_paths))]
        
        comparison_results = {}
        
        for model_path, model_name in zip(model_paths, model_names):
            try:
                model = self.load_model(model_path)
                results = self.evaluate_single_model(model, test_dataloader, model_name)
                comparison_results[model_name] = results
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
                continue
        
        # Create comparison summary
        summary = self._create_comparison_summary(comparison_results)
        
        return {
            'individual_results': comparison_results,
            'summary': summary
        }
    
    def _create_comparison_summary(self, results: Dict[str, Dict]) -> Dict[str, Any]:
        """Create summary comparison of model results"""
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc', 'sensitivity', 'specificity']
        
        summary = {
            'model_count': len(results),
            'best_models': {},
            'metric_comparison': {}
        }
        
        for metric in metrics:
            values = {name: result[metric] for name, result in results.items()}
            best_model = max(values, key=values.get)
            summary['best_models'][metric] = {
                'model': best_model,
                'value': values[best_model]
            }
            summary['metric_comparison'][metric] = values
        
        return summary
    
    def evaluate_continual_learning(self, ewc_trainer: EWCTrainer, 
                                  task_dataloaders: List[DataLoader]) -> Dict[str, Any]:
        """
        Evaluate continual learning performance
        
        Args:
            ewc_trainer: Trained EWC trainer
            task_dataloaders: List of dataloaders for each task
            
        Returns:
            Continual learning evaluation results
        """
        print("Evaluating continual learning performance...")
        
        # Evaluate on all tasks
        task_results = ewc_trainer.evaluate_all_tasks(task_dataloaders, nn.CrossEntropyLoss(), self.device)
        
        # Get summary statistics
        summary_stats = ewc_trainer.get_summary_stats()
        
        # Calculate forgetting measures
        forgetting_measures = self._calculate_forgetting_measures(task_results)
        
        cl_results = {
            'task_results': task_results,
            'summary_stats': summary_stats,
            'forgetting_measures': forgetting_measures
        }
        
        return cl_results
    
    def _calculate_forgetting_measures(self, task_results: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate forgetting measures for each task"""
        forgetting = {}
        
        for task_name, result in task_results.items():
            # Simplified forgetting: difference from ideal performance
            # In practice, this would compare current performance to peak performance
            ideal_performance = 95.0  # Assuming 95% as ideal
            current_performance = result['accuracy']
            forgetting[task_name] = max(0, ideal_performance - current_performance)
        
        return forgetting
    
    def generate_report(self, output_dir: str = "evaluation_results"):
        """
        Generate comprehensive evaluation report
        
        Args:
            output_dir: Directory to save report files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save results as JSON
        with open(output_path / "evaluation_results.json", 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Generate plots
        self._generate_plots(output_path)
        
        # Generate text report
        self._generate_text_report(output_path)
        
        print(f"Evaluation report saved to {output_path}")
    
    def _generate_plots(self, output_path: Path):
        """Generate evaluation plots"""
        if not self.results:
            print("No results to plot")
            return
        
        # Plot 1: Model comparison
        plt.figure(figsize=(15, 10))
        
        # Metrics to plot
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
        model_names = list(self.results.keys())
        
        for i, metric in enumerate(metrics, 1):
            plt.subplot(2, 3, i)
            values = [self.results[name][metric] for name in model_names]
            bars = plt.bar(model_names, values, alpha=0.7)
            plt.title(f'{metric.replace("_", " ").title()}')
            plt.ylabel('Score')
            plt.xticks(rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_path / "model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Confusion matrices
        if len(self.results) <= 4:  # Limit to avoid too many subplots
            fig, axes = plt.subplots(1, len(self.results), figsize=(15, len(self.results) * 4))
            
            for idx, (model_name, result) in enumerate(self.results.items()):
                cm = np.array(result['confusion_matrix'])
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
                axes[idx].set_title(f'{model_name} Confusion Matrix')
                axes[idx].set_xlabel('Predicted')
                axes[idx].set_ylabel('Actual')
            
            plt.tight_layout()
            plt.savefig(output_path / "confusion_matrices.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def _generate_text_report(self, output_path: Path):
        """Generate text report"""
        with open(output_path / "evaluation_report.txt", 'w') as f:
            f.write("Seizure Prediction Model Evaluation Report\n")
            f.write("=" * 50 + "\n\n")
            
            for model_name, result in self.results.items():
                f.write(f"Model: {model_name}\n")
                f.write("-" * 30 + "\n")
                f.write(f"Accuracy: {result['accuracy']:.4f}\n")
                f.write(f"Precision: {result['precision']:.4f}\n")
                f.write(f"Recall: {result['recall']:.4f}\n")
                f.write(f"F1-Score: {result['f1_score']:.4f}\n")
                f.write(f"AUC-ROC: {result['auc_roc']:.4f}\n")
                f.write(f"Sensitivity: {result['sensitivity']:.4f}\n")
                f.write(f"Specificity: {result['specificity']:.4f}\n")
                f.write(f"Avg Inference Time: {result['avg_inference_time']:.6f}s\n")
                f.write(f"Total Samples: {result['total_samples']}\n")
                f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate seizure prediction models")
    parser.add_argument('--mode', choices=['single', 'compare', 'cv', 'cl'], required=True,
                       help='Evaluation mode')
    parser.add_argument('--model_path', help='Path to model checkpoint (for single mode)')
    parser.add_argument('--model_paths', nargs='+', help='Paths to model checkpoints (for compare mode)')
    parser.add_argument('--model_name', default='eegnet', help='Model architecture name')
    parser.add_argument('--data_dir', required=True, help='Directory containing data files')
    parser.add_argument('--labels_csv', required=True, help='CSV file with labels')
    parser.add_argument('--output_dir', default='evaluation_results', help='Output directory')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of CV folds')
    
    args = parser.parse_args()
    
    evaluator = ModelEvaluator()
    
    # Create test data loader
    test_dataset = EEGDataset(Path(args.data_dir), Path(args.labels_csv))
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_batch)
    
    if args.mode == 'single':
        if not args.model_path:
            print("Model path required for single mode")
            return
        
        model = evaluator.load_model(args.model_path, args.model_name)
        results = evaluator.evaluate_single_model(model, test_loader, args.model_name)
        print(f"Results: {results}")
        
    elif args.mode == 'compare':
        if not args.model_paths:
            print("Model paths required for compare mode")
            return
        
        results = evaluator.compare_models(args.model_paths, test_loader)
        print(f"Comparison results: {results['summary']}")
        
    elif args.mode == 'cv':
        results = evaluator.cross_validate_model(args.model_name, args.data_dir, 
                                               args.labels_csv, args.n_folds)
        print(f"Cross-validation results: {results}")
    
    # Generate report
    evaluator.generate_report(args.output_dir)


if __name__ == "__main__":
    main()
