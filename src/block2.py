import random


def run():
    rng = random.Random(102)
    n = 100000
    error_chances = [0.01, 0.03, 0.10]
    count_errors = 0
    errors_by_server = [0, 0, 0]

    # Спочатку обираємо сервер, потім визначаємо наявність збою.
    for _ in range(n):
        value = rng.random()
        if value < 0.50:
            server = 0
        elif value < 0.80:
            server = 1
        else:
            server = 2

        if rng.random() < error_chances[server]:
            count_errors += 1
            errors_by_server[server] += 1

    probability_error = 0.50 * 0.01 + 0.30 * 0.03 + 0.20 * 0.10
    priors = [0.50, 0.30, 0.20]
    rows = [{"event": "P(F)", "theory": probability_error,
             "frequency": count_errors / n, "numerator": count_errors, "denominator": n}]

    # Для умовної частоти знаменник — лише запити зі збоєм.
    for server in range(3):
        frequency = None
        if count_errors > 0:
            frequency = errors_by_server[server] / count_errors
        rows.append({"event": f"P(S{server + 1}|F)",
                     "theory": priors[server] * error_chances[server] / probability_error,
                     "frequency": frequency, "numerator": errors_by_server[server],
                     "denominator": count_errors})

    print("\nБЛОК 2. Варіант 1. Локалізація джерела збоїв")
    print(f"N = {n}; запитів зі збоєм: {count_errors}")
    for row in rows:
        row["absolute_error"] = None
        if row["frequency"] is not None:
            row["absolute_error"] = abs(row["frequency"] - row["theory"])
            print(f"{row['event']:<12} теорія = {row['theory']:.6f}; "
                  f"частота = {row['frequency']:.6f}; похибка = {row['absolute_error']:.6f}")
        else:
            print(f"{row['event']}: умовна частота не визначена, збоїв немає.")

    return rows


if __name__ == "__main__":
    run()
