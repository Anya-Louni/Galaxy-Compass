# Galaxies Without a Compass: Learning Under Rotational Symmetry

Rotation-equivariant convolutional networks on **Galaxy10 DECaLS**, with a
self-supervised atlas of all 17,736 galaxies, latent interpolation between
morphology classes, and a reconstruction-plus-isolation anomaly search.

Galaxy morphology carries no preferred orientation. A spiral photographed at
any position angle belongs to the same class. Most morphology classifiers
address this by showing the network rotated copies during training. This
project encodes the symmetry in the architecture instead, using
E(2)-steerable convolutions, and measures what the encoding is worth under a
controlled comparison.

The result is a statement about label efficiency.

---

## Result

Test accuracy, mean and standard deviation over 3 seeds, on a held-out split
of 3,549 galaxies. Both models perform identical convolutional work and
receive identical augmentation.

| Training labels | C8-steerable | Dense CNN | Difference |
|---|---|---|---|
| 5% (620 images) | 0.6930 ± 0.0150 | 0.6366 ± 0.0363 | +5.6 pp |
| 10% | 0.7611 ± 0.0151 | 0.7218 ± 0.0213 | +3.9 pp |
| 20% | 0.7893 ± 0.0102 | 0.7560 ± 0.0072 | +3.3 pp |
| 50% | 0.8116 ± 0.0038 | 0.7730 ± 0.0061 | +3.9 pp |
| 100% | **0.8217 ± 0.0047** | 0.7745 ± 0.0040 | +4.7 pp |

**The steerable network reaches the dense CNN's all-label accuracy at about
14% of the labels, a label-efficiency factor near 7.** At full supervision it
leads by 4.7 points while carrying 11.8 times fewer parameters.

The crossing fraction falls on the steep portion of the curve, where a small
vertical shift moves it considerably in the horizontal direction. It is
therefore reported per seed as well:

| Metric | Per-seed crossing | Mean | Label efficiency |
|---|---|---|---|
| Accuracy | 16.5%, 16.7%, 9.6% | 14.3% | 7.5x, spanning 6.0x to 10.4x |
| Macro-F1 | 16.9%, 19.6%, 9.7% | 15.4% | 7.1x, spanning 5.1x to 10.3x |

The supported summary is a range from roughly 6x to 10x, centred near 7x.

**A claim independent of that interpolation.** Holding seed and label
fraction fixed and comparing matched pairs, the steerable model wins **15 of
15** comparisons on accuracy and 15 of 15 on macro-F1, giving an exact
two-sided sign test probability of **6.1 x 10⁻⁵**. Every architecture
comparison in the study, at every label budget and every seed, resolved in
the same direction.

**The all-label target sits on a plateau.** Both curves flatten between 50%
and 100% of labels, by 0.10 and 0.15 percentage points respectively. A
baseline still improving at the end of training would understate the labels
it requires, and the crossing fraction would inherit that error.

**Where the variance concentrates.** The dense CNN's spread at 5% labels
(± 0.0363) is roughly triple its spread elsewhere, driven by one seed scoring
0.6785 against 0.6137 and 0.6176. Low-label training of the non-equivariant
model is less stable, which forms part of the result and also means the 5%
row describes a wide distribution.

---

## What is held fixed

Comparisons of this kind fail when the baseline is quietly handicapped. Here
the baseline receives every advantage that matters:

| | C8-steerable | Dense CNN |
|---|---|---|
| Channel widths, kernels, downsampling | identical | identical |
| Convolutional MACs per image | **75.4 M** | **75.4 M** |
| Trainable parameters | 33,286 | **393,514** (11.8x more) |
| Rotation and reflection augmentation | applied | **applied, identical** |
| Optimisation budget | fixed step count | fixed step count |
| Measured CPU throughput | 217 img/s | 233 img/s |

Both models are told about the symmetry in the strongest form a data
pipeline offers. The experiment measures whether being told matches being
built that way.

The steerable model holds fewer parameters because weight sharing across the
group is the mechanism that produces equivariance. It performs the same
number of operations because `e2cnn` expands its steerable basis into a dense
filter bank and calls the same convolution the baseline calls.

Runs share a fixed number of gradient steps rather than a fixed number of
epochs. Under an epoch budget, a run on all the labels would receive twenty
times more updates than a run on 5%, which would confound label efficiency
with optimisation budget.

---

## Verified equivariance

Equivariance is a property that is simple to claim and simple to break.
`scripts/verify_equivariance.py` measures it:

| Measurement | Result |
|---|---|
| Single steerable layer, exact grid rotations | `5.5e-07` max relative error |
| Assembled network, representation drift under a quarter turn | `9.4e-08` |
| Same measurement, dense CNN | `2.6e-02` |

Both figures are taken at initialisation, with random weights and no
augmentation. The invariance is structural.

Evaluated on trained models across all 30 checkpoints, using the four exact
90-degree rotations, which are lossless pixel permutations on an odd frame:

| | Mean accuracy spread | Worst |
|---|---|---|
| C8-steerable | **exactly 0** | exactly 0 |
| Dense CNN | 0.00580 | 0.00958 |

The steerable network assigns every one of the 3,549 test galaxies the same
class at 0, 90, 180 and 270 degrees, in each of its 15 checkpoints. The dense
CNN moves by 0.58 percentage points on average after training with
full-circle rotation augmentation.

No ratio is quoted for that comparison. The denominator is exactly zero, and
the identity is the stronger statement.

### Why the input is 65 pixels wide

A steerable convolution is equivariant to the precision of the arithmetic.
Stride-2 subsampling introduces a second condition. On an even grid the
centre of rotation falls between pixels, the retained lattice maps to a
shifted lattice under a quarter turn, and the discrepancy compounds with
depth. An odd grid fixes the centre pixel and preserves the lattice.

`scripts/probe_pooling.py` sweeps the choice and measures the consequence:

| Input | Pooling | End-to-end drift under a quarter turn |
|---|---|---|
| 64 | antialiased blur-pool | `4.7e-02` |
| 63 | antialiased blur-pool | `5.3e-02` |
| 64 | 2x2 average | `9.6e-08` |
| **65** | **antialiased blur-pool** | **`8.8e-08`** |

Blur-pool at 65 pixels was selected. It is the correct equivariant
subsampling operator for the continuous group, and it retains a 5 by 5 final
feature map.

---

## Averaging over the orbit

An invariant prediction is available without a steerable architecture.
Evaluating an ordinary network on every rotation of the input and averaging
the probabilities produces an exactly invariant output, at a cost of one
forward pass per group element. If that closes the gap, the contribution of
architectural equivariance reduces to a choice about where to spend compute.

Every checkpoint was therefore re-scored with predictions averaged over the
eight element dihedral orbit: four quarter turns and their reflections, each
a lossless permutation of pixels. Both architectures received the identical
treatment.

| Test accuracy, 100% labels | Single pass (1x) | C4-TTA (4x) | D4-TTA (8x) |
|---|---|---|---|
| Dense CNN | 0.7745 | 0.7767 | 0.7781 |
| C8-steerable | 0.8217 | **0.8217** | 0.8247 |

**Orbit averaging returns +0.36 percentage points to the dense CNN for eight
times the inference cost.** The separation between architectures measures
+4.72 points with single passes and +4.67 points with both models averaged.
Inference-time averaging removes the variation across orientations and leaves
the difference in representation quality that training under the constraint
produces.

Label efficiency holds under each framing:

| Comparison | Crossing | Efficiency |
|---|---|---|
| vs single-pass dense CNN | 13.9% | 7.2x |
| vs D4-TTA dense CNN, which pays 8x inference | 15.2% | 6.6x |
| both models given the same 8x budget | 12.4% | **8.1x** |

Averaging the steerable model over the four rotations reproduces its
single-pass accuracy to every printed digit at all five label fractions
(0.692965, 0.761059, 0.789330, 0.811590, 0.821734). Averaging leaves a
prediction unchanged when that prediction is already the same at every
rotation, so this reproduces the exactly-zero spread result through an
independent code path on real test data.

The summary: an invariant prediction is available from test-time averaging,
and accuracy together with label efficiency requires the architecture.

---

## Data

Galaxy10 DECaLS (Leung and Bovy, `astroNN`), Zenodo record **10845026**:
17,736 colour cutouts in DECaLS g, r and z, labelled into ten morphology
classes from aggregated Galaxy Zoo votes.

* Preprocessing verifies the source file by **SHA-256** and asserts the ten
  **published per-class counts** before writing any derived array, so a
  truncated, corrupted or substituted file halts the pipeline.
* Images are cropped from 256 to 224 pixels, stored at 95, and presented to
  the network at 65. The stored margin keeps rotation augmentation well
  posed: a 95 pixel frame rotated through any angle covers the central 65
  pixel field with observed sky, since 95 divided by the square root of two
  is 67.2. A tighter crop admits undefined corners whose orientation a
  network can read in place of the morphology.
* The crop is set from measurement. The stacked radial surface brightness
  profile of all 17,736 galaxies places **96.9%** of the light above sky
  inside the crop, with the profile at the crop edge at **1.4%** of its
  central value.
* Stratified splits under a fixed seed: **12,415 / 1,772 / 3,549**
  (train / validation / test).
* 92 galaxies (0.52%) carry no measured redshift. They are exported with an
  explicit sentinel and displayed as "not measured".

### Augmentation

Rotation, reflection and small translations are applied freely, since each is
an exact symmetry of the labelling function.

Colour is left untouched. The three channels are photometric bands, and their
ratios encode stellar population age, dust content and redshift. Brightness,
contrast and hue jitter are standard on natural images and would destroy that
signal. The pipeline perturbs only with additive noise, which models the
detector and the sky background.

---

## Self-supervised atlas

The same steerable encoder was trained under a SimCLR objective on the
training split alone, for 39 epoch-equivalents, with no labels at any point.
C8 contains rotations and excludes reflections, so parity remains a genuine
augmentation with a real gradient while rotation invariance arrives from the
architecture.

| Measurement | Value |
|---|---|
| k-NN probe accuracy (k=20, held-out test) | **67.2%** |
| k-NN probe macro-F1 | 0.6404 |
| k-means cluster purity against Galaxy Zoo labels | **41.1%** |
| Adjusted mutual information | 0.2731 |
| UMAP trustworthiness (k=15) | 0.9218 |

A representation built without labels supports a k-nearest-neighbour
classifier at 67.2%, against 82.2% for the fully supervised model, placing it
close to what supervision delivers from 5% of the labels. Cluster purity of
41.1% stands against a majority-class floor of 14.9%. The two purest clusters
are the edge-on classes, the morphologies least ambiguous to a human
classifier; the spiral subclasses distribute across clusters, matching where
Galaxy Zoo voters disagree most.

---

## Anomaly ranking

Two signals are combined. Reconstruction error from a convolutional VAE
responds to structure the model never learned to express, and rises with
brightness, size and noise. Local isolation, the mean cosine distance to the
twenty nearest neighbours in the self-supervised embedding, responds to
objects lacking close analogues, and rises along the sparse tail of common
classes. Their rank correlation is **0.264**, so the combined percentile rank
draws on two largely independent views.

Class enrichment in the top 100 against the parent catalogue:

| Class | Found | Expected | Enrichment |
|---|---|---|---|
| Merging | 18 | 10.4 | **1.72x** |
| Barred Spiral | 19 | 11.5 | 1.65x |
| Unbarred Tight Spiral | 15 | 10.3 | 1.45x |
| Round Smooth | 7 | 14.9 | 0.47x |
| In-between Round Smooth | 3 | 11.4 | **0.26x** |

Mergers and structurally complex spirals dominate the candidate list, and
smooth ellipticals fall to the bottom, which follows from their being the
most reconstructible and most densely packed morphology in the embedding.
Cigar Shaped Smooth, the rarest class in the catalogue at 334 members, is
also depleted at 0.53x, which distinguishes the ranking from one that
recovers rare classes.

Every candidate carries its sky coordinates and links to the DESI Legacy
Survey viewer, so the list can be checked against the survey imaging. The
output is written for inspection; membership indicates distance from the
training distribution, which includes artefacts, blends and bright foreground
stars.

---

## Relation to prior work

The closest published work is Pandya, Patel, O and Blazek, *E(2) Equivariant
Neural Networks for Robust Galaxy Morphology Classification*
([arXiv:2311.01500](https://arxiv.org/abs/2311.01500), NeurIPS ML4PS 2023).
They report **95.52 ± 0.18%** test accuracy on Galaxy10 DECaLS with a
D16-steerable network against **84.84 ± 0.14%** for a CNN baseline, trained
for 100 epochs on four NVIDIA A100-80GB GPUs.

Two points of contrast:

1. Their absolute accuracy is far above anything reported here. This project
   trains small networks at 65 pixels on a laptop CPU and does not compete on
   absolute accuracy.
2. Their study varies group order and noise robustness at full supervision.
   The label-fraction curve is the contribution here, and it fills a gap that
   study leaves open.

Foundational references: Cohen and Welling, *Group Equivariant Convolutional
Networks* (ICML 2016); Weiler and Cesa, *General E(2)-Equivariant Steerable
CNNs* (NeurIPS 2019); Dieleman, Willett and Dambre, *Rotation-invariant
convolutional neural networks for galaxy morphology prediction* (MNRAS 2015).

---

## Limitations

* **Scale.** Small networks at 65 pixels on CPU. Absolute accuracy sits well
  below the published state of the art on this dataset. The claim concerns
  the shape of the label-efficiency curve under a controlled comparison.
* **Model selection.** The validation set is held at 10% of the data for
  every label fraction. A strictly realistic low-label protocol would shrink
  it in proportion. Holding it fixed flatters the low-fraction points in
  absolute terms, identically for both architectures.
* **Seeds.** Error bars are one standard deviation over three seeds,
  sufficient to separate the architectures and too few for a tight interval.
* **Group coverage.** Only C8 was trained. Dihedral groups and other orders
  remain untested here, and the literature reports gains from both.
* **Label quality.** Galaxy Zoo labels are aggregated human votes carrying
  real disagreement, concentrated between tight and loose spirals and among
  disturbed galaxies. That disagreement bounds every accuracy figure above.
* **Self-supervised budget.** The contrastive encoder saw 39
  epoch-equivalents, short by the standards of the method. The reported probe
  accuracy is a floor.

---

## Repository layout

```
src/
  data.py              source verification, cropping, stratified splits
  models.py            C8-steerable network and the matched dense baseline
  augment.py           batched rigid transforms and the colour policy
  train_supervised.py  one label-fraction run
  simclr.py            self-supervised contrastive training
  embed.py             encoding, k-NN probe, cluster purity, UMAP
  vae.py               convolutional VAE for latent geometry and reconstruction
  interpolate.py       latent paths and the classifier response along them
  anomaly.py           reconstruction error combined with local isolation
scripts/
  verify_equivariance.py   correctness and cost audit; run this first
  probe_pooling.py         the sweep that selected 65 pixels
  inspect_data.py          validates the crop against the light profile
  run_sweep.py             drives the label-fraction study
  aggregate_sweep.py       headline figure, per-seed crossing, paired test
  rotation_robustness.py   accuracy across the four exact grid rotations
  tta_baseline.py          test-time augmentation over the D4 orbit
  tta_matched_cost.py      matched-inference-cost reduction of those results
  rotation_demo.py         precomputes the rotating-galaxy figure
  run_downstream.py        chains every stage after the sweep
  export_web.py            packages results into web assets
  check_assets.py          numerically verifies atlas indexing
  build_artifact.py        inlines the site into one self-contained file
web/
  index.html           the static site; no build step, no dependencies
```

---

## Reproducing

```bash
pip install -r requirements.txt

mkdir -p data/raw
curl -L -o data/raw/Galaxy10_DECals.h5 \
  "https://zenodo.org/records/10845026/files/Galaxy10_DECals.h5?download=1"

python -m src.data                      # verify checksum, build splits
python scripts/verify_equivariance.py   # equivariance and cost audit
python scripts/inspect_data.py          # crop geometry against light profile

python scripts/run_sweep.py --seeds 0 1 2
python scripts/run_downstream.py
python scripts/tta_baseline.py && python scripts/tta_matched_cost.py

python scripts/export_web.py && python scripts/check_assets.py
python -m http.server 8099 --directory web
```

Every stage runs on CPU. No GPU is required at any point. The full pipeline
takes roughly 12 hours on a four-core laptop, dominated by the 30-run
supervised sweep.

---

## Attribution

Imaging from the **DESI Legacy Imaging Surveys**. Morphology labels from
**Galaxy Zoo**. Dataset packaged as **Galaxy10 DECaLS** by Henry Leung and Jo
Bovy, Zenodo record 10845026. Please cite those sources for any use of the
data.
