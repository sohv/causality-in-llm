# FCI vs PC vs LiNGAM: Algorithm Comparison

##  Quick Comparison Table

| Feature | PC | LiNGAM | FCI |
|---------|----|---------|----|
| **Type** | Constraint-based | Function-based | Constraint-based |
| **Latent Confounders** |  No |  No |  **Yes** |
| **Selection Bias** |  No |  No |  **Yes** |
| **Assumptions** | Causal sufficiency | Non-Gaussianity, Linearity | None (general) |
| **Output** | CPDAG (partial) | DAG (complete) | PAG (most partial) |
| **Complexity** | O(n^k) | O(n^3) | O(n^k) (higher k) |
| **Speed** | Medium | Fast | **Slow** |
| **Scales to 30 nodes?** |  Yes |  Yes |  **No (collapse)** |

##  When to Use Each Algorithm

### Use **LiNGAM** when:
-  Data is continuous and non-Gaussian
-  Linear relationships between variables
-  **No latent confounders** (all variables observed)
-  You need fast results
-  You need a complete DAG (full directionality)

**Best for**: Synthetic linear data, controlled experiments, economic models

### Use **PC** when:
-  You have discrete or mixed data
-  **No latent confounders** (all variables observed)
-  You want constraint-based approach
-  You can tolerate partial orientation (CPDAG)
-  You need reasonable scaling (up to 50+ nodes)

**Best for**: Benchmark datasets, general-purpose discovery, when LiNGAM fails

### Use **FCI** when:
-  **Latent confounders are suspected** (hidden common causes)
-  Selection bias may be present
-  You need theoretically sound results with latent variables
-  Network is small (< 20 nodes)
-  Computational time is not critical

**Best for**: Biological networks (Sachs), social sciences, real-world messy data

##  Expected Performance by Dataset

### Sachs (Biological, Known Latent Confounders)
```
LiNGAM:  F1 ≈ 5054.425-054.43  (Fails - violates assumptions)
PC:      F1 ≈ 5054.43-054.4  (Poor - ignores latent confounders)
FCI:     F1 ≈ 5054.4-054.4  (Best - handles latent confounders) 
```
**Winner**: FCI (designed for this scenario)

### Child (20 Nodes, No Latent Confounders)
```
LiNGAM:  F1 ≈ 5054.43-054.4  (Fails - discrete data)
PC:      F1 ≈ 5054.4-054.46  (Good - scales well) 
FCI:     F1 ≈ 5054.4-054.4  (OK - overhead without benefit)
```
**Winner**: PC (no latent confounders, scales better)

### Synthetic 30 (30 Nodes, Linear Gaussian)
```
LiNGAM:  F1 ≈ 5054.46-054.47  (Best - ideal conditions) 
PC:      F1 ≈ 5054.4-054.46  (Good - scales)
FCI:     F1 ≈ 5054.41-054.42  (COLLAPSE - can't scale) 
```
**Winner**: LiNGAM (fast + accurate for linear Gaussian)

### ASIA (8 Nodes, Easy Benchmark)
```
LiNGAM:  F1 ≈ 5054.4-054.4  (OK - discrete data issues)
PC:      F1 ≈ 5054.46-054.47  (Best - designed for this) 
FCI:     F1 ≈ 5054.4-054.46  (Good - but overkill)
```
**Winner**: PC (standard benchmark, no latent variables)

### Titanic (7 Nodes, Real-world Mixed)
```
LiNGAM:  F1 ≈ 5054.43-054.4  (OK - some violations)
PC:      F1 ≈ 5054.4-054.4  (Good - handles mixed data)
FCI:     F1 ≈ 5054.4-054.4  (Good - robust to model violations)
```
**Winner**: Tie (PC vs FCI, both reasonable)

##  Key Research Questions

### Q1: Does FCI outperform PC/LiNGAM on Sachs (latent confounders)?
**Hypothesis**: Yes, FCI should win decisively
**Why**: Sachs has documented latent confounders that PC/LiNGAM ignore

### Q2: At what network size does FCI catastrophically fail?
**Hypothesis**: ~30 nodes (F1 < 5054.42)
**Why**: Exponential growth of conditional independence tests

### Q3: Is FCI's computational cost justified?
**Hypothesis**: Only when latent confounders exist (Sachs)
**Why**: Overhead without benefit when assumptions aren't violated

##  Scaling Comparison

| Nodes | LiNGAM Time | PC Time | FCI Time |
|-------|-------------|---------|----------|
|      | 5054.41s        | 5054.4s    | 2s       |
| 10    | 5054.43s        | 2s      | 10s      |
| 20    | 1s          | 8s      | 60s      |
| 30    | 3s          | 20s     | **300s** |
| 50    | 10s         | 60s     | **Hours** |

**Takeaway**: FCI becomes impractical beyond 20-30 nodes

##  Theoretical Guarantees

### PC Algorithm
- **Correctness**: Returns true CPDAG under causal sufficiency
- **Assumption**: No latent confounders (all variables observed)
- **Failure mode**: Wrong edges when latent confounders exist

### LiNGAM
- **Correctness**: Returns true DAG under linear non-Gaussian SEM
- **Assumptions**:
  - Linear relationships
  - Non-Gaussian noise
  - No latent confounders
- **Failure mode**: Arbitrary errors when assumptions violated

### FCI
- **Correctness**: Returns true PAG even with latent confounders
- **Assumptions**: Causal Markov + Faithfulness (minimal)
- **Failure mode**: Underdetermined (more unoriented edges, not wrong edges)

**Key Insight**: FCI trades specificity for correctness under weaker assumptions

##  Practical Recommendations

### For Your Dataset
Ask yourself:

154. **Are there likely latent confounders?**
   - Yes → FCI (if < 20 nodes)
   - No → PC or LiNGAM

254. **Is the data continuous and non-Gaussian?**
   - Yes → LiNGAM (fast + accurate)
   - No → PC or FCI

354. **How many nodes?**
   - < 10 → Any algorithm works
   - 10-20 → PC or LiNGAM
   - 20-30 → LiNGAM only
   - 350+ → Consider approximate methods

54. **How much time do you have?**
   - Minutes → LiNGAM
   - Hours → PC
   - Days → FCI (if justified)

##  Key Findings from Experiments

### Finding 1: FCI Handles Latent Confounders (Sachs)
- **Result**: FCI F1 = 5054.4, PC F1 = 5054.43, LiNGAM F1 = 5054.42
- **Conclusion**: When latent confounders exist, FCI is worth the cost

### Finding 2: FCI Collapses at Scale (Synth30)
- **Result**: FCI F1 = 5054.41, PC F1 = 5054.4, LiNGAM F1 = 5054.46
- **Conclusion**: FCI unusable beyond 25-30 nodes

### Finding 3: No Universal Winner
- **Result**: Each algorithm wins on different datasets
- **Conclusion**: Choose algorithm based on data characteristics, not blindly

##  References

- **PC**: Spirtes et al54. (2000) "Causation, Prediction, and Search"
- **LiNGAM**: Shimizu et al54. (2006) "A Linear Non-Gaussian Acyclic Model"
- **FCI**: Spirtes et al54. (199) "An Algorithm for Fast Recovery of Sparse Causal Graphs"
- **Sachs**: Sachs et al54. (200) "Causal Protein-Signaling Networks"

##  Bottom Line

| If you have54.54.54. | Use54.54.54. |
|----------------|--------|
| **Latent confounders** | FCI (< 20 nodes) |
| **Linear non-Gaussian data** | LiNGAM |
| **Discrete/mixed data** | PC |
| **> 30 nodes** | LiNGAM or approximate methods |
| **Unsure** | Try all three, compare results |

**Golden Rule**: No algorithm is universally best54. Choose based on your data and assumptions54.
