import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run():
    root = Path(__file__).resolve().parent.parent
    graphics = root / "graphics"
    graphics.mkdir(exist_ok=True)
    rng = random.Random(101)

    # Теоретичні ймовірності ознаки A та атаки за наявності A.
    probability_a = 0.90 * 0.01 + 0.08 * 0.25 + 0.02 * 0.90
    probability_attack = 0.02 * 0.90 / probability_a
    rows = []
    print("\nБЛОК 1. Байєсівське оновлення для WAF/IDS")
    print(f"P(A) = {probability_a:.6f}; P(H3|A) = {probability_attack:.6f}")
    print("N          M(A)    M(H3 і A)   W(A)       W(H3|A)   Похибка")

    for n in [100, 1000, 10000, 100000, 1000000]:
        count_a = 0
        count_attack_a = 0

        # Для кожного N генерується окрема вибірка запитів.
        for _ in range(n):
            value = rng.random()
            if value < 0.90:
                category = 1
                chance_a = 0.01
            elif value < 0.98:
                category = 2
                chance_a = 0.25
            else:
                category = 3
                chance_a = 0.90

            if rng.random() < chance_a:
                count_a += 1
                if category == 3:
                    count_attack_a += 1

        frequency_a = count_a / n
        frequency_attack = None
        error_attack = None
        if count_a > 0:
            frequency_attack = count_attack_a / count_a
            error_attack = abs(frequency_attack - probability_attack)

        rows.append({
            "N": n, "count_a": count_a, "count_attack_a": count_attack_a,
            "frequency_a": frequency_a, "theory_a": probability_a,
            "error_a": abs(frequency_a - probability_a),
            "frequency_attack_given_a": frequency_attack,
            "theory_attack_given_a": probability_attack,
            "error_attack_given_a": error_attack,
        })
        if frequency_attack is None:
            print(f"{n:<10} Ознака A не виникла; умовна частота не визначена.")
        else:
            print(f"{n:<10} {count_a:<7} {count_attack_a:<11} "
                  f"{frequency_a:.6f}   {frequency_attack:.6f}   {error_attack:.6f}")

    plt.figure(figsize=(8, 4.4))
    plt.plot([row["N"] for row in rows],
             [row["frequency_attack_given_a"] for row in rows],
             "o-", label="Статистична частота W(H3|A)")
    plt.axhline(probability_attack, color="darkred", linestyle="--",
                label=f"Теорія P(H3|A) = {probability_attack:.6f}")
    plt.xscale("log")
    plt.xlabel("Кількість запитів N")
    plt.ylabel("Ймовірність / відносна частота")
    plt.title("Збіжність оцінки ймовірності атаки за ознакою A")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(graphics / "block1_convergence.png", dpi=200)
    plt.close()
    return rows


if __name__ == "__main__":
    run()
