# Dataset Notes

## Synthetic Inventory Risk Dataset

**This dataset is entirely synthetic.** It was generated programmatically for model development and evaluation because no suitable public dataset exists with the exact fields required: `currentStock`, `averageMonthlySales`, `recentSales`, `daysUntilExpiry`, and `risk` (wastage risk label).

**This is not real pharmacy data.** It must not be described as real, anonymized, or sourced from any pharmacy, hospital, or public repository (including Kaggle).

---

## Generation Parameters

- **Seed**: 42 (fixed for reproducibility)
- **Default row count**: 4,000
- **Feature order** (CSV columns):
  1. `currentStock`
  2. `averageMonthlySales`
  3. `recentSales`
  4. `daysUntilExpiry`
  5. `risk`

---

## Labelling Rule (Domain Heuristic)

```
salesRate = 0.7 * averageMonthlySales + 0.3 * recentSales
expectedConsumption = salesRate * (daysUntilExpiry / 30)

risk = HIGH    if currentStock > 2 * expectedConsumption
risk = MEDIUM  if expectedConsumption < currentStock <= 2 * expectedConsumption
risk = LOW     otherwise
```

---

## Noise Method

Bounded label noise is applied near decision boundaries to avoid perfect separability and simulate real-world ambiguity:

- **Near HIGH/MEDIUM boundary** (`|currentStock - 2*expected| / expected < 0.25`): 35% chance to flip between HIGH and MEDIUM
- **Near LOW/MEDIUM boundary** (`|currentStock - expected| / expected < 0.25`): 30% chance to flip between LOW and MEDIUM

This yields approximately **88–94% agreement** between the rule and final labels, not 100%.

---

## Target Class Balance

- **LOW**: 35–45%
- **MEDIUM**: 25–35%
- **HIGH**: 25–35%

---

## Realism Measures

- `daysUntilExpiry`: 1–365, skewed toward shorter horizons (60% sampled from 1–90)
- `averageMonthlySales`: Poisson(λ=6), clipped to [0, 50]
- `recentSales`: Correlated with `averageMonthlySales` (normal multiplier, σ=0.3), clipped to [0, 60]
- `currentStock`: Log-normal multiplier around `expectedConsumption` (mean=0.3, σ=0.7), clipped to [0, 500]
- No duplicate-row deduplication needed; continuous multipliers and Poisson draws avoid excessive identical rows
- Values are not suspiciously round (no systematic multiples of 10/100)

---

## Validation Asserts

The generator enforces:
- No null values
- No negative values in numeric columns
- Risk labels ∈ {LOW, MEDIUM, HIGH}
- Exact column order as specified

---

## Intended Use

This dataset is for **prototyping, baseline modelling, and educational demonstration** only. It encodes a simplified domain heuristic with controlled noise. It does not capture real pharmacy dynamics (seasonality, supplier lead times, batch tracking, regulatory constraints, etc.). Any model trained on this should be validated on real inventory data before deployment.