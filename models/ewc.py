"""
Elastic Weight Consolidation (EWC) for Continual Learning
Implementation based on: "Overcoming catastrophic forgetting in neural networks" (Kirkpatrick et al., 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
from copy import deepcopy


class EWC:
    """Elastic Weight Consolidation for continual learning"""
    
    def __init__(self, model: nn.Module, importance: float = 1000.0):
        """
        Initialize EWC
        
        Args:
            model: Neural network model
            importance: Lambda parameter controlling EWC regularization strength
        """
        self.model = model
        self.importance = importance
        self.fisher_information = {}
        self.optimal_params = {}
        self.previous_tasks = []
        
    def compute_fisher_information(self, dataloader, criterion, device='cpu', num_samples=None):
        """
        Compute Fisher Information Matrix
        
        Args:
            dataloader: DataLoader for current task data
            criterion: Loss function
            device: Device to run computation on
            num_samples: Number of samples to estimate Fisher (None = use all)
        """
        self.model.eval()
        fisher_info = {}
        
        # Initialize Fisher information dict
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher_info[name] = torch.zeros_like(param.data)
        
        sample_count = 0
        total_samples = len(dataloader) if num_samples is None else num_samples
        
        print(f"Computing Fisher Information Matrix using {total_samples} samples...")
        
        for batch_idx, (data, targets) in enumerate(dataloader):
            if num_samples is not None and batch_idx >= num_samples:
                break
            
            data, targets = data.to(device), targets.to(device)
            
            # Forward pass
            output = self.model(data)
            loss = criterion(output, targets)
            
            # Backward pass to get gradients
            self.model.zero_grad()
            loss.backward()
            
            # Accumulate squared gradients (Fisher information)
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_info[name] += param.grad.data ** 2
            
            sample_count += 1
            
            if (batch_idx + 1) % 50 == 0:
                print(f"Processed {batch_idx + 1}/{total_samples} batches...")
        
        # Average Fisher information
        for name in fisher_info:
            fisher_info[name] /= sample_count
        
        self.fisher_information = fisher_info
        
        # Store optimal parameters
        self.optimal_params = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        
        print("Fisher Information Matrix computation completed.")
    
    def ewc_loss(self) -> torch.Tensor:
        """
        Compute EWC regularization loss
        
        Returns:
            EWC regularization loss
        """
        ewc_loss = 0.0
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.fisher_information:
                # EWC penalty: lambda/2 * Fisher * (param - optimal_param)^2
                ewc_loss += (self.fisher_information[name] * 
                           (param - self.optimal_params[name]) ** 2).sum()
        
        return (self.importance / 2) * ewc_loss
    
    def consolidate_task(self, dataloader, criterion, device='cpu'):
        """
        Consolidate knowledge from current task
        
        Args:
            dataloader: DataLoader for current task data
            criterion: Loss function
            device: Device to run computation on
        """
        print(f"Consolidating task {len(self.previous_tasks) + 1}...")
        
        # Compute Fisher information for current task
        self.compute_fisher_information(dataloader, criterion, device)
        
        # Store task info
        self.previous_tasks.append({
            'fisher_info': deepcopy(self.fisher_information),
            'optimal_params': deepcopy(self.optimal_params)
        })
        
        print(f"Task {len(self.previous_tasks)} consolidated successfully.")
    
    def get_regularization_loss(self) -> torch.Tensor:
        """
        Get total EWC regularization loss from all previous tasks
        
        Returns:
            Total EWC regularization loss
        """
        if not self.previous_tasks:
            return torch.tensor(0.0)
        
        total_loss = 0.0
        
        for task_info in self.previous_tasks:
            task_loss = 0.0
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in task_info['fisher_info']:
                    task_loss += (task_info['fisher_info'][name] * 
                                (param - task_info['optimal_params'][name]) ** 2).sum()
            total_loss += task_loss
        
        return (self.importance / 2) * total_loss


class EWCTrainer:
    """Trainer with EWC support for continual learning"""
    
    def __init__(self, model: nn.Module, ewc_importance: float = 1000.0):
        """
        Initialize EWC Trainer
        
        Args:
            model: Neural network model
            ewc_importance: Lambda parameter for EWC
        """
        self.model = model
        self.ewc = EWC(model, importance=ewc_importance)
        self.task_accuracies = []
        self.task_losses = []
        
    def train_task(self, dataloader, val_dataloader, optimizer, criterion, 
                  epochs: int, device='cpu', consolidate: bool = True):
        """
        Train model on a single task with optional EWC consolidation
        
        Args:
            dataloader: Training data loader
            val_dataloader: Validation data loader
            optimizer: Optimizer
            criterion: Loss function
            epochs: Number of epochs
            device: Device to train on
            consolidate: Whether to consolidate after training
            
        Returns:
            Task training history
        """
        task_history = {'losses': [], 'accuracies': []}
        
        print(f"Training task {len(self.task_accuracies) + 1}...")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (data, targets) in enumerate(dataloader):
                data, targets = data.to(device), targets.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(data)
                
                # Standard task loss
                task_loss = criterion(outputs, targets)
                
                # Add EWC regularization loss if we have previous tasks
                ewc_loss = self.ewc.get_regularization_loss()
                total_loss = task_loss + ewc_loss
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += task_loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, targets in val_dataloader:
                    data, targets = data.to(device), targets.to(device)
                    outputs = self.model(data)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
            
            train_acc = 100. * correct / total
            val_acc = 100. * val_correct / val_total
            
            task_history['losses'].append(epoch_loss / len(dataloader))
            task_history['accuracies'].append(val_acc)
            
            print(f'Epoch {epoch+1}/{epochs}: '
                  f'Train Loss: {epoch_loss/len(dataloader):.4f}, '
                  f'Train Acc: {train_acc:.2f}%, '
                  f'Val Loss: {val_loss/len(val_dataloader):.4f}, '
                  f'Val Acc: {val_acc:.2f}%')
        
        # Store task accuracy
        final_accuracy = task_history['accuracies'][-1]
        self.task_accuracies.append(final_accuracy)
        
        # Consolidate knowledge if requested
        if consolidate:
            self.ewc.consolidate_task(dataloader, criterion, device)
        
        print(f'Task {len(self.task_accuracies)} completed with final accuracy: {final_accuracy:.2f}%')
        
        return task_history
    
    def evaluate_all_tasks(self, task_dataloaders, criterion, device='cpu'):
        """
        Evaluate model on all learned tasks
        
        Args:
            task_dataloaders: List of dataloaders for each task
            criterion: Loss function
            device: Device to evaluate on
            
        Returns:
            Dictionary of task accuracies
        """
        self.model.eval()
        task_results = {}
        
        for task_idx, dataloader in enumerate(task_dataloaders):
            correct = 0
            total = 0
            loss = 0.0
            
            with torch.no_grad():
                for data, targets in dataloader:
                    data, targets = data.to(device), targets.to(device)
                    outputs = self.model(data)
                    loss += criterion(outputs, targets).item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            
            accuracy = 100. * correct / total
            avg_loss = loss / len(dataloader)
            
            task_results[f'task_{task_idx + 1}'] = {
                'accuracy': accuracy,
                'loss': avg_loss
            }
            
            print(f'Task {task_idx + 1}: Accuracy = {accuracy:.2f}%, Loss = {avg_loss:.4f}')
        
        return task_results
    
    def plot_learning_progress(self, save_path: str = None):
        """
        Plot learning progress across tasks
        
        Args:
            save_path: Path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 5))
            
            # Plot 1: Task accuracies
            plt.subplot(1, 2, 1)
            task_numbers = list(range(1, len(self.task_accuracies) + 1))
            plt.bar(task_numbers, self.task_accuracies, color='skyblue', alpha=0.7)
            plt.xlabel('Task Number')
            plt.ylabel('Final Accuracy (%)')
            plt.title('Final Accuracy per Task')
            plt.xticks(task_numbers)
            plt.grid(True, alpha=0.3)
            
            # Plot 2: Forgetting measure (if multiple tasks)
            if len(self.task_accuracies) > 1:
                plt.subplot(1, 2, 2)
                # Calculate forgetting measure
                forgetting_measures = []
                for i in range(len(self.task_accuracies)):
                    if i == 0:
                        forgetting = 0  # First task has no forgetting measure
                    else:
                        # Simplified forgetting: difference from peak performance
                        forgetting = max(0, max(self.task_accuracies[:i+1]) - self.task_accuracies[i])
                    forgetting_measures.append(forgetting)
                
                plt.bar(task_numbers, forgetting_measures, color='salmon', alpha=0.7)
                plt.xlabel('Task Number')
                plt.ylabel('Forgetting Measure')
                plt.title('Catastrophic Forgetting Measure')
                plt.xticks(task_numbers)
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Learning progress plot saved to {save_path}")
            else:
                plt.show()
                
        except ImportError:
            print("Matplotlib not available. Skipping plot generation.")
    
    def get_summary_stats(self) -> Dict:
        """
        Get summary statistics of continual learning performance
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.task_accuracies:
            return {}
        
        stats = {
            'num_tasks': len(self.task_accuracies),
            'average_accuracy': np.mean(self.task_accuracies),
            'std_accuracy': np.std(self.task_accuracies),
            'min_accuracy': np.min(self.task_accuracies),
            'max_accuracy': np.max(self.task_accuracies),
            'final_accuracy': self.task_accuracies[-1] if self.task_accuracies else 0
        }
        
        # Calculate backward transfer (if multiple tasks)
        if len(self.task_accuracies) > 1:
            # Simplified backward transfer: compare first task performance
            initial_performance = self.task_accuracies[0]
            # This would need actual evaluation on first task after learning subsequent tasks
            # For now, we'll use a placeholder
            stats['backward_transfer'] = 0  # Placeholder
        
        return stats
