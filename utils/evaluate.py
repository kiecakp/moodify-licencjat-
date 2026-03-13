import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


CLASSES = ['angry', 'happy', 'relaxed', 'sad']


# ==============================================================================
# EWALUACJA JEDNEGO MODELU
# ==============================================================================
def evaluate_model(model, X_tfidf_test, X_vader_test, X_bert_test, y_test):
    """Zwraca dict z wynikami: y_pred, acc, cm."""
    y_pred = model.predict(X_tfidf_test, X_vader_test, X_bert_test)
    acc = np.mean(y_pred == np.array(y_test))
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    return {'y_pred': y_pred, 'acc': acc, 'cm': cm, 'model': model}


def print_best_report(results, y_test):
    """Drukuje pełny raport klasyfikacji dla najlepszego modelu."""
    best_key = max(results, key=lambda k: results[k]['acc'])
    best = results[best_key]
    model = best['model']
    print(f"\n📊 Najlepszy model: {model.name} | tryb={model.mode} | acc={best['acc']:.3f}")
    print(classification_report(y_test, best['y_pred'], target_names=model.classes_))


# ==============================================================================
# WIZUALIZACJE
# ==============================================================================
def save_visualizations(results, results_dir):
    """
    Zapisuje wykresy porównujące wszystkie modele.

    results: dict {klucz_modelu: {'acc', 'cm', 'y_pred', 'y_test', 'model'}}
    """
    os.makedirs(results_dir, exist_ok=True)

    _plot_accuracy_comparison(results, results_dir)
    _plot_confusion_matrix_best(results, results_dir)
    _plot_f1_per_class(results, results_dir)


def _plot_accuracy_comparison(results, results_dir):
    keys = list(results.keys())
    accs = [results[k]['acc'] for k in keys]
    colors = plt.cm.tab10.colors

    # Pokoloruj słupki według modelu (SVM, LR, RF, NB)
    model_colors = {
        'SVM': '#3498DB', 'LogisticRegression': '#E74C3C',
        'RandomForest': '#2ECC71', 'NaiveBayes': '#F39C12',
    }
    bar_colors = [model_colors.get(results[k]['model'].name, '#888') for k in keys]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(keys, accs, color=bar_colors, width=0.6)

    # Linia bazowa random (4 klasy)
    ax.axhline(y=0.25, color='gray', linestyle='--', alpha=0.7, label='Random baseline (0.25)')

    ax.set_ylabel('Accuracy')
    ax.set_title('Porównanie modeli — Accuracy (angry / happy / relaxed / sad)')
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.tick_params(axis='x', rotation=35)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=8)

    # Legenda kolorów modeli
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=name) for name, c in model_colors.items()]
    ax.legend(handles=legend_elements + [
        plt.Line2D([0], [0], color='gray', linestyle='--', label='Random baseline (0.25)')
    ])

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'accuracy_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: accuracy_comparison.png")


def _plot_confusion_matrix_best(results, results_dir):
    best_key = max(results, key=lambda k: results[k]['acc'])
    best = results[best_key]
    model = best['model']
    cm = best['cm']
    classes = model.classes_

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, normalize in zip(axes, [False, True]):
        if normalize:
            cm_plot = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt, cmap, suffix = '.2f', 'YlOrRd', '(Normalized)'
        else:
            cm_plot = cm
            fmt, cmap, suffix = 'd', 'Blues', '(Counts)'

        sns.heatmap(cm_plot, annot=True, fmt=fmt, cmap=cmap,
                    xticklabels=classes, yticklabels=classes, ax=ax)
        ax.set_title(f'Najlepszy model: {model.name}/{model.mode} {suffix}\nacc={best["acc"]:.3f}')
        ax.set_ylabel('True')
        ax.set_xlabel('Predicted')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'confusion_matrix_best.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: confusion_matrix_best.png")


def _plot_f1_per_class(results, results_dir):
    classes = list(results[next(iter(results))]['model'].classes_)
    keys = list(results.keys())
    x = np.arange(len(classes))
    width = 0.8 / len(keys)

    model_colors = {
        'SVM': '#3498DB', 'LogisticRegression': '#E74C3C',
        'RandomForest': '#2ECC71', 'NaiveBayes': '#F39C12',
    }

    fig, ax = plt.subplots(figsize=(14, 5))

    for i, key in enumerate(keys):
        r = results[key]
        model = r['model']
        report = classification_report(
            r['y_test'], r['y_pred'],
            target_names=classes,
            output_dict=True,
            zero_division=0,
        )
        f1s = [report[cls]['f1-score'] for cls in classes]
        color = model_colors.get(model.name, '#888')
        ax.bar(x + i * width, f1s, width, label=key, color=color, alpha=0.85)

    ax.set_xticks(x + width * (len(keys) - 1) / 2)
    ax.set_xticklabels(classes)
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 per klasa — wszystkie modele')
    ax.legend(fontsize=7, ncol=2)
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'f1_per_class.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: f1_per_class.png")
