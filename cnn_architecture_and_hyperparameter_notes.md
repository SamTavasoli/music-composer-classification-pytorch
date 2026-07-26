# CNN Architecture and Hyperparameter Search Notes

**Composer Classification from MIDI Files, Group 6**

Companion document to [`composer_classification.ipynb`](composer_classification.ipynb), Sections 9, 10, and 11.

## Contents

1. [Input Representation](#1-input-representation)
2. [Layer-by-Layer Breakdown](#2-layer-by-layer-breakdown)
3. [Training Settings Held Fixed During the Search](#3-training-settings-held-fixed-during-the-search)
4. [CNN Hyperparameter Search: What Each Setting Does](#4-cnn-hyperparameter-search-what-each-setting-does)
5. [LSTM Hyperparameter Search (For Comparison)](#5-lstm-hyperparameter-search-for-comparison)
6. [Parameter Count Summary](#6-parameter-count-summary)

---

## 1. Input Representation

Every MIDI file enters the network as a sequence of 4,096 notes. Each note carries four values:

| Feature | Description |
| --- | --- |
| Encoded pitch | An integer token from 1 to 128 (MIDI pitch + 1), with 0 reserved for padding |
| Velocity | How hard the note was struck |
| Duration | How long the note is held, in seconds |
| Time interval | The time since the previous note started (inter-onset interval) |

The three continuous features are standardized with means and standard deviations computed from the training set only.

The pitch token is not fed to the network as a raw integer. It passes through an embedding layer, `nn.Embedding(129, 32, padding_idx=0)`, which maps each of the 128 possible pitches to a learned 32-dimensional vector. An embedding is used instead of one-hot encoding for two reasons. First, a one-hot vector treats every pair of pitches as equally different, but musically some pitches behave similarly (notes an octave apart, notes inside the same register), and a learned embedding can place such pitches near each other in the vector space. Second, 32 dense dimensions are cheaper to convolve over than 128 sparse ones. The `padding_idx=0` argument pins the padding token to a zero vector and excludes it from gradient updates, so padded positions never inject learned content into the network.

The 32 embedding dimensions are concatenated with the 3 continuous features, giving a 35-channel signal of length 4,096. The tensor is then permuted to `(batch, 35, 4096)`, the layout `nn.Conv1d` expects: channels are feature types and the single spatial axis is time, measured in note positions.

---

## 2. Layer-by-Layer Breakdown

The network is three convolutional blocks followed by global max pooling, dropout, and a linear classifier. Total: **106,916 trainable parameters**.

| Stage | Sequence length | Channels |
| --- | --- | --- |
| Input | 4,096 | 35 |
| Block 1 | 1,024 | 64 |
| Block 2 | 256 | 128 |
| Block 3 | 256 | 128 |
| Global max pool | 1 | 128 |
| Classifier | - | 4 |

### Block 1

```python
Conv1d(35 -> 64, kernel_size=5, padding=2) + BatchNorm1d + ReLU + MaxPool1d(4)
```

Sequence length 4,096 to 1,024. Parameters: 11,264 (conv) + 128 (batch norm).

- The convolution slides 64 independent filters over the sequence. Each filter is a 35 x 5 template: it looks at 5 consecutive notes across all 35 input channels and produces a high activation wherever the local pattern matches. At this depth the filters learn primitive musical events: short interval shapes, ornament-like figures, repeated-note cells, and characteristic velocity or timing patterns. Padding 2 keeps the output the same length as the input, so no notes at the edges are lost.
- `BatchNorm1d` normalizes each of the 64 channels over the batch and time axes to zero mean and unit variance, then rescales with learned parameters. This keeps activation magnitudes stable regardless of how the previous layer drifts during training, which is what lets the network tolerate the relatively large 0.001 learning rate. The batch-statistics noise also acts as a mild regularizer.
- `ReLU` provides the nonlinearity. Without it, three stacked convolutions would collapse into one linear filter. ReLU also gives sparse activations and does not saturate for positive inputs, so gradients flow cleanly.
- `MaxPool1d(4)` keeps only the strongest activation in every window of 4 positions, shrinking the sequence from 4,096 to 1,024. This does three things: it makes the representation tolerant to a motif shifting by a few notes, it cuts the compute of every later layer by 4x, and it widens the time span that the next layer's kernel covers.

### Block 2

```python
Conv1d(64 -> 128, kernel_size=5, padding=2) + BatchNorm1d + ReLU + MaxPool1d(4)
```

Sequence length 1,024 to 256. Parameters: 41,088 (conv) + 256 (batch norm).

- This convolution no longer sees raw notes; it sees block 1 pattern activations. A kernel of width 5 here covers 24 notes of the original sequence (the receptive field after the first pooling), so block 2 composes the primitive figures into phrase-level patterns: sequences of motifs, accompaniment textures, cadence-like shapes.
- The channel count doubles from 64 to 128 while the resolution drops. This is the standard pyramid trade in convolutional networks: as the network looks at longer time spans, there are more distinct pattern combinations worth representing, so the pattern vocabulary grows while the sequence gets shorter.

### Block 3

```python
Conv1d(128 -> 128, kernel_size=3, padding=1) + BatchNorm1d + ReLU
```

Sequence length stays 256. Parameters: 49,280 (conv) + 256 (batch norm).

- A final refinement stage with a narrow kernel and no pooling. It mixes neighboring phrase-level activations without discarding any more resolution. After this block, each position in the 256-step feature map has a receptive field of **68 consecutive notes** of the original piece, so the deepest features respond to patterns roughly the length of a full musical phrase.
- Keeping the channel count at 128 rather than doubling again caps the parameter budget. This block is already the most expensive one in the network.

### Global max pooling

```python
AdaptiveMaxPool1d(1)   # (batch, 128, 256) -> (batch, 128), no parameters
```

- For each of the 128 learned patterns, this keeps a single number: the strongest activation of that pattern anywhere in the piece. The feature vector therefore answers 128 questions of the form "does this composer-typical pattern occur in this piece, and how strongly", regardless of where in the piece it occurs.
- This choice is what makes zero padding harmless at the classifier stage. Padded regions produce weak activations, and a maximum simply ignores weak values. Average pooling, by contrast, would dilute the signal of a short piece with thousands of padded positions.
- It also decouples the classifier from sequence length: whatever the length of the feature map, the output is always 128 numbers.

### Dropout

Rate 0.30 in the baseline, 0.50 in the final model.

- During training, each of the 128 pooled features is zeroed with the given probability and the survivors are rescaled. The classifier therefore cannot rely on any single pattern detector and must spread evidence across many of them. Placing dropout directly before the classifier attacks overfitting at the point where memorization of training pieces would be easiest.

### Classifier

```python
Linear(128 -> 4)   # 516 parameters
```

- A single linear layer maps the 128-dimensional piece summary to four logits, one per composer. `CrossEntropyLoss` applies log-softmax to these logits internally. Only 516 of the 106,916 parameters sit in the classifier; nearly all capacity is spent on pattern extraction, which is intentional. A wide fully connected head over the un-pooled feature map (128 x 256 = 32,768 inputs) would have added millions of parameters and overfit the 1,117 training files immediately. Global pooling plus a tiny head avoids that failure mode.

### Why this architecture suits the task

- Convolutions process all 4,096 positions in parallel, so one epoch takes about 0.3 seconds on GPU. The LSTM must walk the sequence step by step and cannot parallelize over time.
- Composer identity in this dataset is carried largely by local style: ornament shapes, texture, harmonic rhythm, voice-leading figures. A bank of motif detectors with "did it occur anywhere" pooling captures exactly that. The price is that note order beyond the 68-note receptive field and large-scale form are invisible to the model. The test results (78.33% CNN against 70.83% LSTM) suggest local patterns matter more here than long-range structure.

---

## 3. Training Settings Held Fixed During the Search

The Section 10 search deliberately froze everything except one setting per run, so the effect of each change could be attributed cleanly.

- **Optimizer: Adam.** Keeps a per-parameter adaptive step size built from running estimates of the gradient mean and variance. It is a robust default for both architectures; switching to SGD would have required a separate learning-rate search of its own.
- **Batch size: 32.** Batch size and learning rate interact, since larger batches give less noisy gradients and usually tolerate larger steps, so changing both at once would blur which change caused an effect.
- **Epoch budget: 20, with best-epoch checkpointing.** After every epoch the model is scored on the validation set, and the state dict with the highest validation accuracy (ties broken by lower loss) is kept. This is early stopping in effect: even if later epochs overfit, the saved model comes from the epoch before that damage.
- **Gradient clipping at norm 1.0.** Caps the size of any single update. Mainly a safeguard for the LSTM, since recurrent networks can produce exploding gradients, but applied uniformly so training dynamics stay comparable.
- **Random seed 511, reset before every run.** Every configuration starts from the same initialization and sees the same batch order, so result differences come from the tuned setting, not from luck of the draw.
- **Search strategy: one factor at a time.** Five LSTM runs and four CNN runs instead of a combinatorial grid. Cost grows linearly with the number of settings, and each result has an unambiguous cause. The limitation is that interactions between settings are never observed.

---

## 4. CNN Hyperparameter Search: What Each Setting Does

All runs start from the Section 9 baseline: learning rate 0.001, 64 filters, dropout 0.30, unweighted loss, validation accuracy 85.83% at epoch 7.

| Configuration | Change | Best epoch | Validation accuracy | Outcome |
| --- | --- | --- | --- | --- |
| Baseline | - | 7 | 85.83% | Reference |
| Lower learning rate | 0.001 to 0.0005 | 10 | 84.58% | Worse |
| More filters | 64 to 96 | 6 | 85.42% | No gain |
| Higher dropout | 0.30 to 0.50 | 17 | **86.25%** | **Winner** |
| Class-weighted loss | Inverse frequency weights | 13 | 85.83% | No gain |

### Learning rate: 0.001 to 0.0005

The learning rate scales every Adam update. A smaller step gives smoother, more stable convergence and less bouncing around a minimum, at the cost of speed. Here the halved rate simply had not converged far enough inside the fixed 20-epoch budget; its best epoch (10) scored 1.25 points below the baseline.

**Effect on performance:** controls the speed and stability trade-off. Too low underfits within a fixed budget, too high oscillates or diverges.

### Number of filters: 64 to 96

The filter count is the size of the pattern vocabulary in each block, and the deeper blocks scale with it (192 channels instead of 128). More filters mean more distinct motifs can be detected, but the parameter count roughly doubles, from about 107 thousand to about 226 thousand, and every extra parameter is another opportunity to memorize the training set. The wider model peaked early, at epoch 6, and did not beat the baseline: capacity was not the bottleneck, generalization was.

**Effect on performance:** raises the capacity ceiling. Helps only when the model is underfitting.

### Dropout: 0.30 to 0.50 (winner)

Raising the drop probability makes the classifier train on ever-changing random subsets of the 128 pooled features. The baseline showed the classic overfitting signature, with training accuracy reaching 96% by epoch 20 while validation stalled, so stronger regularization was the targeted fix, and it worked. Notably, the best epoch moved from 7 to 17: heavier dropout slowed down memorization enough that useful learning continued much longer into the budget.

**Effect on performance:** trades raw fitting speed for a smaller train/validation gap. Too much dropout would starve the classifier of evidence and underfit.

### Class-weighted loss

Each class weight is `total_samples / (num_classes * class_count)`, so a misclassified Chopin file costs the model more loss than a misclassified Bach file. The intent is to stop the majority class, Bach at about 63% of files, from dominating the gradient. Accuracy stayed exactly at baseline while validation loss rose from 0.4615 to 0.6396: the reweighting moved errors between classes rather than removing them, and made the model less confident overall.

**Effect on performance:** shifts the precision and recall balance toward minority classes. It does not add information, so it cannot create accuracy that the features do not support.

### Selection

The dropout 0.50 configuration won on validation accuracy (86.25% against 85.83% for the baseline) and became the final CNN, later scoring **78.33% accuracy and 0.7739 weighted F1** on the held-out test set.

A follow-up check, run outside the notebook, trained the winning configuration and four alternatives across three random seeds each:

| Configuration | Mean validation accuracy | Mean train/validation gap |
| --- | --- | --- |
| Dropout 0.50 (final) | 85.97% | +2.4 pts |
| Dropout 0.50 + weight decay 1e-4 | 85.28% | +1.7 pts |
| Dropout 0.50 + weight decay 1e-3 | 84.44% | -0.1 pts |
| Dropout 0.30 + weight decay 1e-4 | 85.56% | +7.6 pts |
| Dropout 0.50 + label smoothing 0.1 | 85.97% | -0.4 pts |

None of the alternatives beat dropout 0.50 on mean validation accuracy. Stronger weight decay shrank the train/validation gap but pushed accuracy down, which is underfitting rather than better generalization. Seed-to-seed spread on identical settings was about 1.3 points, a useful scale for judging which differences in the tables above are real.

---

## 5. LSTM Hyperparameter Search (For Comparison)

All runs start from the Section 8 baseline: learning rate 0.001, hidden size 64, one layer, dropout 0.30, unweighted loss, validation accuracy 75.83%.

| Configuration | Change | Validation accuracy | Outcome |
| --- | --- | --- | --- |
| Baseline | - | 75.83% | Reference |
| Lower learning rate | 0.001 to 0.0005 | 74.17% | Worse |
| Larger hidden size | 64 to 128 | 76.67% | Small gain |
| Two LSTM layers | 1 to 2 | **79.17%** | **Winner** |
| Higher dropout | 0.30 to 0.50 | 76.25% | Small gain |
| Class-weighted loss | Inverse frequency weights | 77.08% | Small gain |

- **Learning rate.** Same mechanism and same outcome as for the CNN: slower convergence did not pay off within 20 epochs.
- **Hidden size.** The hidden size is the width of the state vector the LSTM carries from note to note, that is, how much of the sequence history it can remember at once. Doubling it gave a modest gain, so the baseline was slightly capacity-limited.
- **Number of layers (winner).** Stacked LSTM layers build a hierarchy: the first layer models note-to-note transitions, the second models sequences of the first layer's outputs, which is closer to phrase-level structure. This change also activates the inter-layer dropout that the model class only applies when `num_layers > 1`, so it added depth and regularization together.
- **Dropout.** Applied to the final hidden state before the classifier. Small gain over baseline, much smaller than the effect depth had.
- **Class-weighted loss.** Same mechanism as the CNN version. Here it helped accuracy slightly but raised validation loss to 0.8983, again trading confidence for class balance.

The two winners illustrate a useful contrast: the LSTM was underfitting, so more depth helped most, while the CNN was overfitting, so more regularization helped most. Matching the fix to the actual failure mode, rather than applying the same fix to both models, is what the one-factor search made visible.

---

## 6. Parameter Count Summary

| Layer | Parameters |
| --- | ---: |
| Embedding (129 x 32) | 4,128 |
| Block 1 conv + batch norm | 11,392 |
| Block 2 conv + batch norm | 41,344 |
| Block 3 conv + batch norm | 49,536 |
| Global max pool | 0 |
| Dropout | 0 |
| Classifier (128 -> 4) | 516 |
| **Total** | **106,916** |

For reference, the baseline LSTM has 30,244 parameters and the final two-layer LSTM about 63,500. The CNN spends its larger budget on parallel pattern detectors, which the test results justified: 78.33% accuracy against 70.83% for the LSTM, with far shorter training time per epoch.
