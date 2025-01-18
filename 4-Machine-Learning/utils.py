import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import copy
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score, precision_score
from typing import Dict, List, Tuple, Any

class ModelTrainer:
    def __init__(self, model: nn.Module, criterion: nn.Module, optimizer: optim.Optimizer, scheduler: lr_scheduler._LRScheduler, dataloaders: Dict[str, DataLoader], dataset_sizes: Dict[str, int], device: torch.device) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloaders = dataloaders
        self.dataset_sizes = dataset_sizes
        self.device = device
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        self.best_model_wts = copy.deepcopy(model.state_dict())
        self.best_acc = 0.0
        self.epochs_no_improve = 0

    def train(self, num_epochs: int = 10, patience: int = 3, min_delta: float = 0.0) -> Tuple[nn.Module, Dict[str, List[float]]]:
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}\n{'-' * 10}")

            for phase in ['train', 'val']:
                self.model.train() if phase == 'train' else self.model.eval()
                running_loss, running_corrects = 0.0, 0

                for inputs, labels in self.dataloaders[phase]:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)
                        preds = torch.max(outputs, 1)[1]

                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    self.scheduler.step()

                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]
                self.history[f'{phase}_loss'].append(epoch_loss)
                self.history[f'{phase}_acc'].append(epoch_acc.item())

                print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

                # Early stopping logic during validation
                if phase == 'val':
                    if epoch_acc - self.best_acc > min_delta:
                        self.best_acc = epoch_acc
                        self.best_model_wts = copy.deepcopy(self.model.state_dict())
                        self.epochs_no_improve = 0
                    else:
                        self.epochs_no_improve += 1

            # Check early stopping condition
            if self.epochs_no_improve >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                break

        # Load best weights
        self.model.load_state_dict(self.best_model_wts)
        print("Training complete")
        return self.model, self.history

    def plot_history(self) -> None:
        # Accuracy
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.plot(self.history['train_acc'], label='Train Acc')
        plt.plot(self.history['val_acc'], label='Val Acc')
        plt.title('Accuracy vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # Loss
        plt.subplot(1,2,2)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Val Loss')
        plt.title('Loss vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()

class ModelEvaluator:
    def __init__(self, model: nn.Module, dataloaders: Dict[str, DataLoader], device: torch.device, class_names: List[str], metrics_dir: str) -> None:
        self.model = model
        self.dataloaders = dataloaders
        self.device = device
        self.class_names = class_names
        self.metrics_dir = metrics_dir

    def evaluate_model(self, model_name: str, architecture_details: str) -> Dict[str, Any]:
        self.model.eval()
        all_preds = []
        all_labels = []

        # Collect predictions and labels
        with torch.no_grad():
            for inputs, labels in self.dataloaders['test']:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Metrics
        acc = np.mean(all_preds == all_labels)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        rec = recall_score(all_labels, all_preds, average='weighted')
        prec = precision_score(all_labels, all_preds, average='weighted')

        # Classification report
        report = classification_report(all_labels, all_preds, target_names=self.class_names, output_dict=True)

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)

        # Plot confusion matrix
        plt.figure(figsize=(8, 8))
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.title(f"Confusion Matrix - {model_name}", fontsize=16)
        plt.colorbar()
        tick_marks = np.arange(len(self.class_names))
        plt.xticks(tick_marks, self.class_names, rotation=45, fontsize=12)
        plt.yticks(tick_marks, self.class_names, fontsize=12)

        # Add annotations
        fmt = 'd'
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.xlabel("Predicted Labels", fontsize=14)
        plt.ylabel("True Labels", fontsize=14)
        plt.tight_layout()

        # Save confusion matrix
        confusion_matrix_path = os.path.join(self.metrics_dir, f"{model_name}_confusion_matrix.png")
        plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Plot classification report
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis('off')
        ax.axis('tight')

        # Prepare data for the classification report
        report_data = [
            [class_name, f"{report[class_name]['precision']:.2f}",
             f"{report[class_name]['recall']:.2f}", f"{report[class_name]['f1-score']:.2f}"]
            for class_name in self.class_names
        ]
        report_table = [
            ["Class", "Precision", "Recall", "F1-Score"]
        ] + report_data

        table = ax.table(cellText=report_table, loc='center', cellLoc='center', colLabels=None)
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.2)

        plt.title(f"Classification Report - {model_name}\n{architecture_details}", fontsize=14)
        plt.tight_layout()

        # Save classification report
        classification_report_path = os.path.join(self.metrics_dir, f"{model_name}_classification_report.png")
        plt.savefig(classification_report_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Metrics saved to: {self.metrics_dir}")

        return {
            "accuracy": acc,
            "f1_score": f1,
            "recall": rec,
            "precision": prec,
            "confusion_matrix_path": confusion_matrix_path,
            "classification_report_path": classification_report_path
        }
    
# Wrappers
def train(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs=10, patience=3, min_delta=0.0):
    trainer = ModelTrainer(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device)
    return trainer.train(num_epochs, patience, min_delta)

def plot_history(history):
    trainer = ModelTrainer(None, None, None, None, None, None, None)
    trainer.history = history
    trainer.plot_history()

def evaluate_model(model, dataloaders, device, class_names, model_name, architecture_details):
    evaluator = ModelEvaluator(model, dataloaders, device, class_names, os.path.dirname(__file__))
    return evaluator.evaluate_model(model_name, architecture_details)