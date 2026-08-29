# Methodology

This page documents the statistical methods implemented in `rasch-per`. All
methods are standard, published psychometric techniques used throughout
physics / discipline-based education research (PER / DBER). No proprietary
assessment content or data is included; every example uses synthetic data
from the package's own simulator.

## Classical Test Theory (CTT)

### Item difficulty

Difficulty is the proportion correct per item,

```
p_j = mean(response_j)
```

computed over persons who answered the item (missing responses ignored).

### Item discrimination

Discrimination is the corrected item-total (point-biserial) correlation between
an item and the **rest score** (the total score excluding that item). Bootstrap
standard errors are reported for each item.

### Reliability

- **Cronbach's alpha** - internal-consistency reliability over listwise complete
  cases.
- **McDonald's omega** - single-factor reliability via an eigenvalue
  approximation.
- **Ferguson's delta** - a discriminatory-power index that is population
  dependent.

## Rasch model

The dichotomous Rasch model gives the probability that person `i` answers item
`j` correctly:

```
P(X_ij = 1 | theta_i, beta_j) = exp(theta_i - beta_j) / (1 + exp(theta_i - beta_j))
```

where `theta_i` is person ability and `beta_j` is item difficulty, both on the
same logit scale.

### Estimation

- **MML** (marginal maximum likelihood, the default) integrates a
  `N(0, sigma^2)` ability distribution by Gauss-Hermite quadrature, mirroring
  R's `TAM`. The scale is identified by fixing `mean(beta) = 0` together with
  the fixed population SD `sigma`.
- **JML** (joint maximum likelihood) is also available and can be selected with
  `estimator="JML"`.

### Fit statistics

Per-item **infit** and **outfit** mean-square statistics are computed from the
residuals, with low-stakes and high-stakes flagging presets.

## Dimensionality

Principal Components Analysis of Residuals (PCAR) is used to check the
unidimensionality assumption. The first-contrast eigenvalue is reported; a value
greater than 2.0 suggests a meaningful second dimension (Linacre, 1998).

## Local independence

Yen's Q3 statistic is the pairwise correlation of item residuals. Large positive
Q3 values between a pair of items indicate a violation of local independence
(Yen, 1984).

## Differential Item Functioning (DIF)

Uniform DIF is assessed with Lord's chi-square test:

1. Split the sample into reference and focal groups using a supplied label.
2. Calibrate Rasch item difficulties separately for each group.
3. Place the two calibrations on a common scale with mean/mean linking.
4. Per item, Lord's chi-square (1 df, Wald form) tests
   `H0: beta_ref = beta_focal`.
5. Benjamini-Hochberg controls the false discovery rate across items.
6. The effect size is reported on the ETS delta scale:

   ```
   delta = -2.35 * ln(odds_focal / odds_ref)
   ```

   and binned into ETS categories: **A** (`|delta| < 1.0`, negligible),
   **B** (`1.0 <= |delta| < 1.5`, moderate), **C** (`|delta| >= 1.5`, large).
   A positive delta means the item is easier for the focal group.

## References

- Rasch, G. (1960). *Probabilistic Models for Some Intelligence and Attainment
  Tests*.
- Wright, B. D., & Stone, M. H. (1979). *Best Test Design*.
- Smith, R. M. (2000). Fit analysis in latent trait measurement models.
- Linacre, J. M. (1998). Detecting multidimensionality. *Rasch Measurement
  Transactions*.
- Yen, W. M. (1984). Effects of local item dependence on the fit and equating
  performance of the three-parameter logistic model. *Applied Psychological
  Measurement*.
- Lord, F. M. (1980). *Applications of Item Response Theory to Practical Testing
  Problems*.
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate.
  *Journal of the Royal Statistical Society*.
- Cronbach, L. J. (1951). Coefficient alpha. *Psychometrika*.
- McDonald, R. P. (1999). *Test Theory: A Unified Treatment*.
- Ferguson, G. A. (1949). On the theory of test discrimination. *Psychometrika*.
