# Stock Market Regime Detector


A quantitative finance project that identifies hidden market regimes from
historical price data using a **Gaussian Hidden Markov Model (HMM)** trained
on daily log returns of the S&P 500 ETF (SPY).

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Project Structure](#project-structure)
3. [Theory](#theory)
   - [Hidden Markov Models](#hidden-markov-models)
   - [Why Regimes are Hidden States](#why-regimes-are-hidden-states)
   - [Log Returns](#log-returns)
   - [Transition Probabilities](#transition-probabilities)
4. [How It Works](#how-it-works)
5. [Outputs](#outputs)
6. [Limitations](#limitations)
7. [Future Extensions](#future-extensions)

---

## Quick Start

```bash
# 1. Clone / download the project
cd regime_detector

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline
python main.py                         # default: SPY, 2010-01-01 → today
python main.py --ticker QQQ --start 2015-01-01   # custom
```

Results are written to `outputs/`:

| Path | Contents |
|------|----------|
| `outputs/figures/01_closing_price.png` | SPY price history |
| `outputs/figures/02_price_by_regime.png` | Price coloured by regime |
| `outputs/figures/03_return_histograms.png` | Return distributions |
| `outputs/figures/04_transition_matrix.png` | Transition heatmap |
| `outputs/models/hmm_spy.pkl` | Serialised HMM |
| `outputs/regime_labels_spy.csv` | Date-indexed regime labels |
| `outputs/state_statistics_spy.csv` | Per-regime summary stats |

---

## Project Structure

```
regime_detector/
├── data/
│   └── raw/               ← cached CSV downloads (auto-created)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_regime_detection.ipynb
├── src/
│   ├── data_loader.py     ← yfinance download + caching
│   ├── feature_engineering.py  ← log returns + StandardScaler
│   ├── hmm_model.py       ← HMM training, prediction, labelling
│   ├── visualization.py   ← all four plots
│   └── utils.py           ← logging, path helpers, pretty-print
├── outputs/
│   ├── figures/
│   └── models/
├── requirements.txt
├── README.md
└── main.py                ← end-to-end entry point
```

---

## Theory

### Hidden Markov Models

A **Hidden Markov Model** is a generative probabilistic model that assumes:

1. The system is always in one of *K* unobserved (hidden) states.
2. At each time step the hidden state emits an observable according to a
   state-specific probability distribution.
3. The hidden state at time *t+1* depends **only** on the state at time *t*
   (the Markov property).

Formally an HMM is described by three parameter sets:

| Symbol | Name                  | Meaning                                                            |
| ------ | --------------------- | ------------------------------------------------------------------ |
| **π**  | Initial probabilities | $\mathrm{P}(\text{first state }= k)$                               |
| **A**  | Transition matrix     | $A[i,j] = \mathrm{P}(\text{next state } = j\|\text{ current }= i)$ |
| **B**  | Emission parameters   | $μ_k$, $Σ_k$ of the Gaussian for state $k$                         |

Training uses the **EM algorithm** (Expectation-Maximisation) to
maximise the likelihood of the observed data.  Regime prediction uses the
**Viterbi algorithm** to find the single most-likely hidden state sequence.

### Why Regimes are Hidden States

Real market regimes — bull runs, sideways grinding, crash periods — are
never directly observed.  We see *prices* and derived *returns*, not the
underlying economic state.  A HMM is therefore a natural fit:

* **Hidden states** ≡ market regimes (bear, bull, volatile, …)
* **Observations** ≡ daily log returns
* **Emission model** ≡ each regime has its own characteristic mean return and
  volatility, captured by a Gaussian distribution

### Log Returns

Daily log returns are computed as:
\[
r_t = \log(\text{Close}_t / \text{Close}_{t-1})
\]
Log returns are preferred over simple returns because they are:

* **Time-additive**: multi-day returns sum to the total log return.
* **Approximately normally distributed** for short horizons, matching the
  Gaussian emission assumption of the HMM.
* **Stationary**, unlike raw price levels.

Before training, log returns are standardised to zero mean and unit variance
using `sklearn.preprocessing.StandardScaler`.  This ensures the EM algorithm
converges efficiently regardless of the asset's price level.

### Transition Probabilities

The transition matrix **A** has entries:
$$
A[i, j] = \mathrm{P}(\text{state at } (t+1) = j | \text{ state at } t = i)
$$
Key interpretations:

* **High diagonal values** (e.g. $A[0,0]$ = 0.97) mean regime *i* is **sticky**:
  once the market enters it, it tends to stay there for many days.
* **Off-diagonal values** measure how quickly the market transitions from one
  regime to another.
* **Crash regimes** typically have lower self-transition probabilities —
  crashes are violent but shorter-lived than calm bull markets.

---

## How It Works

```
yfinance download
      │
      ▼
 Closing prices
      │
      │  log(Close_t / Close_{t-1})
      ▼  
 Log returns  (n × 1)
      │
      │  StandardScaler
      ▼
 Scaled returns (n × 1)
      │
      │  GaussianHMM.fit()  [EM, 1000 iterations]
      ▼
 Fitted HMM
      │
      │  GaussianHMM.predict()  [Viterbi]
      ▼
 State sequence  [0, 0, 2, 1, 1, …]
      │
      │  mean / std per state → label_states()
      ▼
 Regime labels  ["Low Vol Bull", "Crash", …]
      │
      │
      ▼
 Visualisations + saved artefacts
```

---

## Outputs

### Regime summary table

| Regime | Mean Return | Volatility | Count | Frequency |
|--------|------------|------------|-------|-----------|
| Low Volatility Bull | +0.00056 | 0.00480 | 2 100 | 56.2% |
| High Volatility Bull | +0.00102 | 0.01140 | 1 200 | 32.1% |
| Crash / Bear Regime | −0.00342 | 0.02780 | 437 | 11.7% |

*(Exact numbers vary with the training window.)*

### Figures

| Figure | Description |
|--------|-------------|
| `01_closing_price.png` | Full SPY price history as a line chart |
| `02_price_by_regime.png` | Same chart with each day coloured by its regime |
| `03_return_histograms.png` | Overlapping density plots per regime, vertical dashed lines at regime means |
| `04_transition_matrix.png` | Annotated heatmap of the 3 × 3 transition matrix |

---

## Limitations

1. **Gaussian emissions** — Real return distributions are fat-tailed.  A
   Student-*t* or mixture emission model would capture crash tails better.
2. **Single feature** — Only log returns are used.  Volatility clustering,
   volume, and macro factors are ignored.
3. **State ordering is arbitrary** — HMMs do not guarantee that state 0 is
   always the bull state.  Labels are derived post-hoc from statistics and
   may shift across re-runs if the random seed or data window changes.
4. **No look-ahead guard** — The full history is used to label past dates.
   In a live trading context this constitutes look-ahead bias.
5. **Stationarity assumption** — The transition probabilities are assumed
   constant.  In reality, regime dynamics shift across macroeconomic cycles.
6. **Fixed number of states** — *K = 3* is a design choice; the optimal
   number of regimes should be selected via information criteria (AIC / BIC)
   or cross-validated log-likelihood.

---

## Future Extensions

### Additional Features
* **Realised volatility** (rolling 5-day or 21-day standard deviation of
  returns) to give the HMM explicit information about the risk environment.
* **Momentum** (e.g. 1-month or 12-month return) to capture trend-following
  dynamics separately from volatility.
* **Autocorrelation** of returns as a feature — negative autocorrelation often
  characterises mean-reverting regimes.
* **Volume / breadth indicators** such as the advance-decline ratio.

### Model Improvements
* **Markov Switching Models** (Hamilton 1989) — a closely related econometric
  framework with richer macroeconomic interpretability.
* **GARCH extensions** — GARCH-HMM hybrids model conditional heteroskedasticity
  within each regime, addressing the fat-tail limitation.
* **Student-*t* HMM** — replacing Gaussian emissions with heavier-tailed
  distributions.
* **Bayesian HMM** — placing priors on transition probabilities to prevent
  regime fragmentation with sparse data.
* **Online / sequential updating** — real-time regime estimation as new data
  arrives (e.g. using the forward algorithm).

### Trading Applications
* **Regime-conditional position sizing** — reduce exposure during detected
  crash regimes.
* **Factor rotation** — tilt between value, momentum, and defensive factors
  based on the active regime.
* **Volatility targeting** — scale position sizes inversely with the emission
  volatility of the current state.
* **Options strategy selection** — sell volatility in low-vol bull regimes,
  buy protection in crash regimes.

---

