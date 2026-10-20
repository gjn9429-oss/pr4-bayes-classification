import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def posterior(features, prior_spam, chances_spam, chances_ham):
    # Добутки враховують як наявність, так і відсутність кожного слова.
    score_spam = prior_spam
    score_ham = 1 - prior_spam
    for index in range(len(features)):
        if features[index] == 1:
            score_spam *= chances_spam[index]
            score_ham *= chances_ham[index]
        else:
            score_spam *= 1 - chances_spam[index]
            score_ham *= 1 - chances_ham[index]
    total = score_spam + score_ham
    if total == 0:
        raise ValueError("Комбінація ознак неможлива за заданою моделлю.")
    return score_spam / total


def calculate_metrics(tp, fp, tn, fn):
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1": f1}


def save_confusion_plot(tp, fp, tn, fn, path, title):
    matrix = [[tn, fp], [fn, tp]]
    fig, ax = plt.subplots(figsize=(5.8, 4.1))
    ax.imshow(matrix, cmap="Blues", vmin=0)
    ax.set_xticks([0, 1], ["Ham", "Spam"])
    ax.set_yticks([0, 1], ["Ham", "Spam"])
    ax.set_xlabel("Прогнозований клас")
    ax.set_ylabel("Фактичний клас")
    ax.set_title(title)
    labels = [["TN", "FP"], ["FN", "TP"]]
    for row in range(2):
        for column in range(2):
            color = "white" if matrix[row][column] > max(tp, fp, tn, fn) / 2 else "black"
            ax.text(column, row, f"{labels[row][column]}\n{matrix[row][column]}",
                    ha="center", va="center", color=color, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run():
    root = Path(__file__).resolve().parent.parent
    graphics = root / "graphics"
    graphics.mkdir(exist_ok=True)
    rng = random.Random(103)
    n = 10000
    prior_spam = 0.30
    chances_spam = [0.80, 0.60]
    chances_ham = [0.05, 0.10]
    tp = fp = tn = fn = 0

    for _ in range(n):
        actual_spam = rng.random() < prior_spam
        chances = chances_spam if actual_spam else chances_ham
        features = []
        # Ознаки незалежні за умови відомого фактичного класу.
        for chance in chances:
            features.append(int(rng.random() < chance))
        probability = posterior(features, prior_spam, chances_spam, chances_ham)
        predicted_spam = probability >= 0.50
        if actual_spam and predicted_spam:
            tp += 1
        elif not actual_spam and predicted_spam:
            fp += 1
        elif not actual_spam and not predicted_spam:
            tn += 1
        else:
            fn += 1

    metrics = calculate_metrics(tp, fp, tn, fn)
    summary = {"N": n, "TP": tp, "FP": fp, "TN": tn, "FN": fn, **metrics}
    print("\nБЛОК 3. Наївний байєсівський класифікатор, дві ознаки")
    print(f"N = {n}; TP = {tp}; FP = {fp}; TN = {tn}; FN = {fn}")
    for name, value in metrics.items():
        print(f"{name} = {value:.6f}")
    save_confusion_plot(tp, fp, tn, fn, graphics / "block3_confusion.png",
                        "Матриця помилок класифікатора, дві ознаки")
    return summary


if __name__ == "__main__":
    run()
