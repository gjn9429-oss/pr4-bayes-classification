import random
from pathlib import Path

from block3 import posterior, calculate_metrics, save_confusion_plot


def run():
    root = Path(__file__).resolve().parent.parent
    graphics = root / "graphics"
    graphics.mkdir(exist_ok=True)
    rng = random.Random(104)
    n = 100000
    prior_spam = 0.20

    # Додаткові параметри навчальної моделі: методичка їх не задає.
    # Перші дві ознаки — free і click з блоку 3, третя — слово win.
    chances_spam = [0.80, 0.60, 0.70]
    chances_ham = [0.05, 0.10, 0.05]
    counts = {}
    spam_counts = {}
    tp = fp = tn = fn = 0

    for _ in range(n):
        actual_spam = rng.random() < prior_spam
        chances = chances_spam if actual_spam else chances_ham
        features = []
        for chance in chances:
            features.append(int(rng.random() < chance))
        pattern = "".join(str(value) for value in features)
        counts[pattern] = counts.get(pattern, 0) + 1
        spam_counts[pattern] = spam_counts.get(pattern, 0) + int(actual_spam)

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

    # Порівнюємо P(Spam|X) із часткою справжнього спаму для кожного X.
    rows = []
    for first in [0, 1]:
        for second in [0, 1]:
            for third in [0, 1]:
                features = [first, second, third]
                pattern = f"{first}{second}{third}"
                probability = posterior(features, prior_spam, chances_spam, chances_ham)
                count = counts.get(pattern, 0)
                count_spam = spam_counts.get(pattern, 0)
                frequency = count_spam / count if count > 0 else None
                error = abs(frequency - probability) if frequency is not None else None
                rows.append({"pattern": pattern, "count": count, "count_spam": count_spam,
                             "theory_spam": probability, "frequency_spam": frequency,
                             "absolute_error": error,
                             "predicted_class": "Spam" if probability >= 0.50 else "Ham"})

    metrics = calculate_metrics(tp, fp, tn, fn)
    summary = {"N": n, "TP": tp, "FP": fp, "TN": tn, "FN": fn, **metrics}
    print("\nБЛОК 4. Варіант 1. Детекція спаму за трьома словами")
    print("Додаткова модель: P(w|Spam) = [0.8, 0.6, 0.7]; P(w|Ham) = [0.05, 0.1, 0.05]")
    print("X     Кількість    P(Spam|X)    W(Spam|X)    Похибка    Прогноз")
    for row in rows:
        if row["frequency_spam"] is not None:
            print(f"{row['pattern']}   {row['count']:<10}  {row['theory_spam']:.6f}     "
                  f"{row['frequency_spam']:.6f}     {row['absolute_error']:.6f}   {row['predicted_class']}")
        else:
            print(f"{row['pattern']}: комбінація у вибірці не виникла.")
    print(f"TP = {tp}; FP = {fp}; TN = {tn}; FN = {fn}")
    for name, value in metrics.items():
        print(f"{name} = {value:.6f}")
    save_confusion_plot(tp, fp, tn, fn, graphics / "block4_confusion.png",
                        "Матриця помилок класифікатора, три ознаки")
    return summary


if __name__ == "__main__":
    run()
