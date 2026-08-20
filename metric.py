import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, fbeta_score, roc_auc_score, average_precision_score, log_loss
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve

class MetricsCalculator():

    @staticmethod
    def calculate_all(y_true, y_pred, y_pred_proba, beta: float = 1.0) -> dict:
        """Считает полный набор метрик для бинарной классификации."""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f_beta': fbeta_score(y_true, y_pred, beta=beta),
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'pr_auc': average_precision_score(y_true, y_pred_proba),
            'log_loss': log_loss(y_true, y_pred_proba)
        }
    
    @staticmethod
    def print_report(y_true, y_pred, y_pred_proba, dataset_name="Test", beta: float = 1.0):
        """Красиво выводит отчет по метрикам в консоль."""
        metrics = MetricsCalculator.calculate_all(y_true, y_pred, y_pred_proba, beta=beta)

        print(f"\n--- Отчет по метрикам ({dataset_name}) ---")
        print(f"Accuracy:     {metrics['accuracy']:.4f}")
        print(f"Precision:    {metrics['precision']:.4f}")
        print(f"Recall:       {metrics['recall']:.4f}")
        print(f"F{beta}-score:    {metrics['f_beta']:.4f}")
        print(f"ROC-AUC:      {metrics['roc_auc']:.4f}")
        print(f"PR-AUC:       {metrics['pr_auc']:.4f}")
        print(f"LogLoss:      {metrics['log_loss']:.4f}")
        print("-" * 30)
        
        return metrics
    
    @staticmethod
    def get_confusion_matrix(y_true, y_pred) -> pd.DataFrame:
        """Возвращает Confusion Matrix в виде таблицы DataFrame."""
        cm = confusion_matrix(y_true, y_pred)
        return pd.DataFrame(
            cm, 
            index=['Факт: 0', 'Факт: 1'],
            columns=['Прогноз: 0', 'Прогноз: 1']
        )
    
    @staticmethod
    def plot_roc_pr_curves(y_true, y_pred_proba):
        """Строит графики ROC-AUC и Precision-Recall кривых в один ряд."""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc_val = auc(fpr, tpr)

        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc_val = auc(recall, precision)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        #ROC Curve
        ax1 = axes[0]
        ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_val:.4f})')
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax1.set_xlim([0, 1.0])
        ax1.set_ylim([0, 1.05])
        ax1.set_xlabel('False Positive Rate', fontsize=12)
        ax1.set_ylabel('True Positive Rate', fontsize=12)
        ax1.set_title('Receiver Operating Characteristic (ROC)', fontsize=14)
        ax1.legend(loc="lower right", fontsize=12)
        ax1.grid(alpha=0.3)

        #PR Curve
        ax2 = axes[1]
        ax2.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc_val:.4f})')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall', fontsize=12)
        ax2.set_ylabel('Precision', fontsize=12)
        ax2.set_title('Precision-Recall Curve', fontsize=14)
        ax2.legend(loc="lower left", fontsize=12)
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_feature_importance(model, feature_names, top_n=15):
        """Визуализирует важность признаков для линейных моделей (коэффициенты) или деревьев/бустингов."""
        if hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
            title = "Важность признаков (Абсолютные коэффициенты LogReg)"
        elif hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            title = "Важность признаков (Feature Importances)"
        else:
            print("У данной модели нет встроенного атрибута для оценки важности признаков.")
            return

        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=True).tail(top_n)

        plt.figure(figsize=(10, max(6, top_n * 0.4)))
        plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue', edgecolor='navy')
        plt.xlabel('Важность', fontsize=12)
        plt.title(title, fontsize=14)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.show()

    staticmethod
    def plot_comparison_curves(models_dict, X_full, X_cat, y):
        """
        Строит совмещенные ROC и PR кривые для нескольких моделей на одном графике,
        автоматически выбирая правильный препроцессинг для каждого типа моделей.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for name, model in models_dict.items():
            if name in ['Tree', 'Random Forest', 'Extra Forest']:
                current_X_test = X_cat
            else:
                current_X_test = X_full

            # Получаем вероятности для класса 1
            if hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(current_X_test)[:, 1]
            else:
                y_pred_proba = model.decision_function(current_X_test)
                
            # 1. ROC Curve
            fpr, tpr, _ = roc_curve(y, y_pred_proba)
            roc_auc_val = auc(fpr, tpr)
            axes[0].plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc_val:.4f})')

            # 2. Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(y, y_pred_proba)
            pr_auc_val = auc(recall, precision)
            axes[1].plot(recall, precision, lw=2, label=f'{name} (PR AUC = {pr_auc_val:.4f})')

        # Оформление ROC графика
        axes[0].plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--')
        axes[0].set_xlim([0.0, 1.0])
        axes[0].set_ylim([0.0, 1.05])
        axes[0].set_xlabel('False Positive Rate', fontsize=12)
        axes[0].set_ylabel('True Positive Rate', fontsize=12)
        axes[0].set_title('Сравнение ROC кривых', fontsize=14)
        axes[0].legend(loc="lower right", fontsize=11)
        axes[0].grid(alpha=0.3)

        # Оформление Precision-Recall графика
        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlabel('Recall', fontsize=12)
        axes[1].set_ylabel('Precision', fontsize=12)
        axes[1].set_title('Сравнение Precision-Recall кривых', fontsize=14)
        axes[1].legend(loc="lower left", fontsize=11)
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.show()