# `relmatch` explained — from intuition to math to code

> A teaching walkthrough for someone new to dataset distillation. We use **one running
> example** (three classes: 🐱 cat, 🐶 dog, 🚗 car) and carry it all the way through:
> the idea, the math, the algorithm (traced by hand), the code, and the diagnostics.
> By the end you should be able to say *what* `relmatch` preserves, *how* the code does it,
> and *what the diagnostic numbers mean*.

---

## 0. TL;DR (read this first, it'll make sense by the end)

RDED builds a tiny training set by keeping the images a pretrained model is **most confident**
about. Those are the "textbook" examples of each class — and precisely because they are textbook,
they throw away the information about **how classes relate to each other** (a cat that looks a bit
like a dog still says "cat", but it *also* whispers "dog").

`relmatch` keeps a small set whose **average opinion of the teacher** reproduces the whole class's
average opinion — including those whispers about other classes. The thing it tries to preserve is a
small table called the **class-relation matrix** `R`. That table is where "cats are confusable with
dogs but not with cars" lives, and that kind of boundary information is what calibration / out-of-
distribution detection / robustness all depend on.

---

## 1. Vocabulary (the five words you need)

- **Teacher.** A neural network already trained on the full dataset. We never change it; we only
  ask it questions.
- **Softmax / soft label.** When the teacher sees an image, it outputs a probability for *every*
  class, and they sum to 1. For a 3-class problem an output might be `[0.80, 0.18, 0.02]` =
  "80% cat, 18% dog, 2% car." That vector is the image's **soft label**. The big number on the
  true class is **confidence**; the small numbers on the *other* classes are the **dark knowledge**
  — the teacher quietly telling you what else the image resembles.
- **Dataset distillation.** Replace a huge training set with a tiny one that still trains a good
  model. **IPC** = "images per class" in that tiny set (e.g. IPC = 10).
- **`stock` (= plain RDED).** The baseline selection rule: for each class, keep the IPC crops the
  teacher is **most confident** about.
- **Selector.** A rule that decides *which* real crops to keep. `relmatch` is a new selector that
  sits next to RDED's existing ones (`stock`, `random`, `covmatch`, `momentmatch`, `qddpp`).

---

## 2. The idea in one picture

Take the cat class. Suppose the full dataset has four cat crops, and the teacher's soft labels
(columns = `[cat, dog, car]`) are:

```
        cat    dog    car
 c1 :  0.90   0.08   0.02     <- a "pure" cat (very confident)
 c2 :  0.80   0.18   0.02
 c3 :  0.70   0.28   0.02
 c4 :  0.60   0.36   0.04     <- a "dog-ish" cat (still a cat, but resembles a dog)
```

We can only keep **2** of them (IPC = 2). Which two?

- **`stock`** keeps the two most confident cats → `c1, c2`. Both are "pure" cats. The distilled set
  now believes cats look *almost only* like cats. The fact that real cats are routinely confused
  with dogs has been **deleted**.
- **`relmatch`** keeps two crops whose **average** looks like the average *real* cat — so it will
  deliberately pair a pure cat with a dog-ish cat. The distilled set then remembers "a cat is mostly
  a cat, but noticeably dog-like."

`relmatch`'s whole job is to keep that second, relational picture. Now let's make "the average real
cat" precise.

---

## 3. The object we care about: the class-relation matrix `R`

**Definition.** For `K` classes, `R` is a `K × K` table where

$$ R[i,j] \;=\; \text{average soft-label mass that class-}i\text{ images put on class-}j, $$

averaged over **all** images of class `i` in the dataset. In plain words: **row `i` is the average
soft label of class `i`.**

Let's build **row "cat"** from the four crops above — just average the columns:

```
cat column:  (0.90 + 0.80 + 0.70 + 0.60) / 4 = 0.750
dog column:  (0.08 + 0.18 + 0.28 + 0.36) / 4 = 0.225
car column:  (0.02 + 0.02 + 0.02 + 0.04) / 4 = 0.025
                                               -----
                              R[cat, :] =  [ 0.750, 0.225, 0.025 ]   (sums to 1.0)
```

Read that row out loud: **"The average cat is 75% cat, 22.5% dog, 2.5% car."** The **diagonal**
entry `R[cat,cat]=0.75` is just self-confidence. The **off-diagonal** entries `[_, 0.225, 0.025]`
are the interesting part — the class's *relational signature*: **cats lean toward dogs, almost never
toward cars.** Do the same for the dog and car rows and you get the full `3×3` matrix `R`.

> Mental model: `R` is the dataset's "confusion fingerprint." The off-diagonal is which classes
> blur into which. That fingerprint is global (it's about the whole dataset), cheap (one average),
> and it's exactly what confidence-greedy selection destroys.

---

## 4. What "good selection" means: `R̂ ≈ R`

The tiny distilled set we keep induces its **own** class-relation matrix, call it `R̂` ("R-hat").
For the cat row, `R̂[cat,:]` is just the average soft label of the **selected** cat crops.

- `stock` keeps `c1,c2` → `R̂[cat,:] = ([0.90+0.80]/2, [0.08+0.18]/2, [0.02+0.02]/2) = [0.85, 0.13, 0.02]`.
  Compare to the target `[0.75, 0.225, 0.025]`: it **overstates** cat-confidence (0.85 vs 0.75) and
  **understates** the dog-resemblance (0.13 vs 0.225). The relational signature is wrong.
- **Goal of `relmatch`:** pick the crops so `R̂[cat,:]` matches `R[cat,:]` as closely as possible.

A crucial simplification makes this tractable: **row `i` of `R̂` depends only on the crops we pick
for class `i`.** So matching the whole matrix `R̂ ≈ R` splits into `K` independent per-class
problems — and "pick crops for one class" is exactly what RDED's selector already does, once per
class. `relmatch` changes *only* the rule for choosing within a class.

---

## 5. The objective (the math, gently)

Fix one class `i` with target row `r = R[i,:]`. Let the candidate crops have soft labels
`p₁, p₂, …` (each a `K`-vector). If we select a set `S` of them, the induced row is their mean:

$$ \hat r(S) \;=\; \frac{1}{|S|}\sum_{s \in S} p_s . $$

We choose `S` (with `|S|` = IPC) to make `r̂(S)` close to `r`:

$$ \boxed{\;J(S) \;=\; \sum_{j=1}^{K} w_j \,\bigl(\hat r(S)_j - r_j\bigr)^2\;}\qquad \text{minimize over } S. $$

That's just **squared error between two vectors**, coordinate by coordinate, with a per-coordinate
weight `wⱼ`. The weight is where the **one knob** lives:

- `wⱼ = 1` for every other class `j ≠ i`, and `w_i = ` **`diag_weight`** for the self-class column.
- **`diag_weight = 0` (the default)** → the self-class column is ignored, so we match **only the
  off-diagonal** (the inter-class signature). This is the "how do cats relate to *other* classes"
  match — the thing the research is about.
- **`diag_weight = 1`** → match the **full row**, self-confidence included (the literal `R`).

**Why is 0 the default?** The diagonal entry (`0.75`-ish) is huge and swings a lot from crop to crop,
while the off-diagonal entries (`0.02`–`0.36`) are small. If you include the diagonal with full
weight, the big self-confidence numbers dominate the squared error and the match becomes "reproduce
the confidence distribution" — drowning out the faint relational signal we actually want. Zeroing it
puts 100% of the objective on the inter-class profile. (You can always turn it back up with the CLI
flag `--relmatch-diag-weight`.)

> Note: this is the **same shape** of objective as the existing `momentmatch` selector — but
> `momentmatch` matches a mean in the network's *feature* space (hundreds of numbers, about visual texture),
> whereas `relmatch` matches a mean in **soft-label space** (`K` numbers, about *class relationships*).
> Same machinery, different — and complementary — target.

---

## 6. The algorithm: greedy forward selection

Minimizing `J(S)` exactly would mean trying every size-IPC subset — far too many. So we use a
**greedy** rule, the same pattern every RDED selector uses:

> Start empty. Repeat IPC times: **add the one crop that makes the running average closest to the
> target.** Never remove.

Let's trace it **by hand** on the cat class, with `diag_weight = 0` (so the weight vector is
`w = [0, 1, 1]` — ignore the cat column, match dog & car). Target `r = [0.75, 0.225, 0.025]`.

**Step 1** (set is empty, so "average if we add crop j" is just crop j itself). Compute the
off-diagonal cost `(p_dog − 0.225)² + (p_car − 0.025)²` for each:

```
 c1 [.90 .08 .02]:  (.08-.225)² + (.02-.025)²  =  .021025 + .000025  =  .02105
 c2 [.80 .18 .02]:  (.18-.225)² + (.02-.025)²  =  .002025 + .000025  =  .00205   <-- smallest
 c3 [.70 .28 .02]:  (.28-.225)² + (.02-.025)²  =  .003025 + .000025  =  .00305
 c4 [.60 .36 .04]:  (.36-.225)² + (.04-.025)²  =  .018225 + .000225  =  .01845
```

Pick **c2**. Running sum `s1 = [0.80, 0.18, 0.02]`.

**Step 2** (now `|S| = 1`, so adding crop j gives average `(s1 + pⱼ)/2`). Cost of the new average:

```
 + c1 -> mean [.85 .13 .02]:  (.13-.225)² + (.02-.025)²  =  .00905
 + c3 -> mean [.75 .23 .02]:  (.23-.225)² + (.02-.025)²  =  .00005   <-- smallest
 + c4 -> mean [.70 .27 .03]:  (.27-.225)² + (.03-.025)²  =  .00205
```

Pick **c3**. Done.

**Result.** `relmatch` selects **`{c2, c3}`**, with induced row
`R̂[cat,:] = [0.75, 0.23, 0.02]`. Compare the off-diagonals:

```
 target        off-diagonal:  dog .225   car .025
 relmatch {c2,c3}          :  dog .230   car .020     <- almost exact
 stock    {c1,c2}          :  dog .130   car .020     <- "forgot" the dog-resemblance
```

`relmatch` reproduced the cat's relational signature; `stock` flattened it. That is the entire
point, shown on four numbers.

> Why greedy is OK: it's fast (one cheap scan per pick), **deterministic** (same input → same
> output, no randomness), and for "match an average" it tends to do the sensible thing — its first
> pick lands near the target, and each later pick *corrects* the running average toward `r` (notice
> step 2 chose `c3`, a slightly dog-heavier crop, to pull the average up to the target's `0.225`).

---

## 7. The code (the core, mapped to the math)

The whole selector is one short greedy loop. Lives in `synthesize/utils.py` as `_select_relmatch`.
Here is the heart of it, annotated against Section 5–6:

```python
def _select_relmatch(n, cand_probs, target_row, diag_weight, diag_idx, gen):
    # cand_probs [P, K] = candidates' soft labels;  target_row [K] = r = R[i,:]
    n  = min(n, cand_probs.shape[0])
    Pc = cand_probs.double()
    P, K = Pc.shape
    r = target_row.double()

    w = torch.ones(K, dtype=torch.float64)        # weight vector w
    if diag_idx is not None and 0 <= int(diag_idx) < K:
        w[int(diag_idx)] = float(diag_weight)     # w_i = diag_weight  (0 by default -> off-diagonal)

    s1   = torch.zeros(K, dtype=torch.float64)    # running SUM of selected soft labels
    avail = torch.ones(P, dtype=torch.bool)       # which candidates are still free
    selected = []
    for step in range(1, n + 1):                  # repeat IPC times
        k    = float(step)
        mean = (s1.unsqueeze(0) + Pc) / k         # [P, K] : the average IF we added each candidate
        diff = mean - r.unsqueeze(0)              # distance of that average from the target r
        J    = ((diff * diff) * w.unsqueeze(0)).sum(dim=1)   # the weighted squared error J  (per candidate)
        J[~avail] = float("inf")                  # don't re-pick something already chosen
        j = int(torch.argmin(J))                  # "add the crop that makes the average closest"
        selected.append(j); avail[j] = False
        s1 += Pc[j]                               # update the running sum
    return torch.tensor(selected[:n], dtype=torch.long)
```

Line-by-line, this *is* the hand trace: `mean` is "the average if we add each candidate", `J` is the
objective `J(S)` from Section 5, `argmin` is "add the best one", `s1 += Pc[j]` is the running-sum
update. The `w` vector is the diagonal knob.

**Where do `cand_probs` and `target_row` come from?** Inside the per-class `selector(...)` (same file),
the teacher already runs on every candidate crop, so we have its logits for free. Two lines turn them
into what the greedy needs:

```python
probs = F.softmax(best_logits, dim=1)     # [pool, K]  soft labels of the class's crops
# ... then in the dispatcher _select_variant(...):
local = _select_relmatch(n, probs[eligible],   # candidates = the realism-floored pool
                            probs.mean(0),      # target r = mean over the WHOLE pool = R[i,:]
                            diag_weight, diag_idx, gen)
```

Two things worth noticing:

1. **The target is the full pool's mean** (`probs.mean(0)`), our best on-hand estimate of the true
   row `R[i,:]`. The candidates are the **realism-floored** subset (`probs[eligible]`).
2. **The realism floor** is a guard shared by all variants: it first throws away crops the teacher
   reads as garbage (very low confidence), *then* lets `relmatch` choose among the believable ones.
   So `relmatch` reaches for the relational signature **without** picking nonsense crops. (`diag_idx`
   is just the column index of the class being processed — the "self-class" coordinate the knob
   refers to.)

That's the entire method. No training, no optimization loop, no randomness.

---

## 8. The diagnostics: did it actually work?

We *claim* `relmatch` preserves `R`. The diagnostics are how we **check** that claim — cheaply,
without training a single student. Two helpers in `validation/diagnostics.py`:

**(a) Build the matrix from any set of images** — `class_relation_matrix(logits, labels, K)`:

```python
probs = softmax(logits)              # soft labels
R = zeros(K, K)
R.index_add_(0, labels, probs)       # sum each class's soft labels into its row
R /= counts                          # divide by #images per class  ->  row i = average soft label
```

Run it on a big batch of **real** images → `R_real` (the truth). Run it on the **distilled** set →
`R̂_syn`. Now compare them.

**(b) Compare the two matrices** — `relation_divergence(R_real, R_syn)` returns four numbers:

| metric | what it measures | good direction |
|---|---|---|
| `rel_frob` | overall distance `‖R_real − R̂‖` (all entries) | ↓ lower |
| `rel_frob_offdiag` | distance using **only off-diagonal** entries (the inter-class part) | ↓ lower **(headline)** |
| `rel_row_cos_offdiag` | `1 − ` average cosine of the off-diagonal **rows** → is the *shape/direction* of each class's signature right (not just its size)? | ↓ lower |
| `rel_eig_overlap` | do the **top eigenvectors** of the two matrices line up? (in `[0,1]`) | ↑ higher |

The first three are flavors of "how far apart are the two fingerprints." `rel_eig_overlap` is the
subtle one: the top eigenvectors of `R` are its **dominant confusion patterns** — e.g. an eigenvector
with weight on both cat and dog encodes "there is a strong cat–dog blur in this dataset." An overlap
near `1` means the distilled set kept the **same dominant confusion structure**; near `0` means it
invented a different one. You don't need the linear algebra to use it: **higher = the big-picture
relational geometry survived.**

**Why we bother.** Three uses, in increasing ambition:

1. **A sanity check / mechanism proof.** It confirms the selector did the thing it promises *before*
   you spend GPU-days training students.
2. **A coverage curve.** Plot `rel_frob_offdiag` as you grow IPC: `stock` saturates early and never
   fills in the off-diagonal; `relmatch` drives it down. That single figure tells the story.
3. **A causal ablation ("degrade-R").** Deliberately make `R̂` worse and watch downstream
   trustworthiness (calibration, OOD) get worse in lockstep — evidence that preserving `R` *causes*
   the gains, rather than just correlating with them.

**The real numbers.** Running this on `cifar100` (teacher `conv3`, IPC = 10) gave:

```
 set        rel_frob   off-diag   row_cos_off   eig_overlap
 stock        3.36       0.786       0.415          0.200
 relmatch     2.89       0.704       0.315          0.352
```

Read it: `relmatch` is **closer** on every distance (lower `frob`, lower off-diagonal, lower row-cos)
**and** its principal confusion axes line up almost twice as well (`eig_overlap` 0.35 vs 0.20). In
words — exactly our toy example at full scale: **`stock` collapses the inter-class geometry,
`relmatch` restores it.**

(The same numbers are printed, and logged to `logs/diagnostics.jsonl`, by
`python tools/diagnose_geometry.py --subset cifar100 --arch-name conv3 --ipc 10 --select-method relmatch`.)

---

## 9. Recap — the whole story in five sentences

1. The teacher's soft labels carry **dark knowledge**: how much each image resembles *other* classes.
2. Average that over a class and you get a row of the **class-relation matrix `R`** — the dataset's
   confusion fingerprint, whose **off-diagonal** is the inter-class signature.
3. RDED's `stock` keeps the most confident crops and so **erases** that fingerprint; `relmatch`
   keeps crops whose **average** reproduces it (`R̂ ≈ R`).
4. It does this with a tiny **greedy** loop — "add the crop that pulls the running average toward the
   target" — repeated IPC times, with a knob (`diag_weight`, default 0) that focuses the match on the
   off-diagonal.
5. The **diagnostics** (`class_relation_matrix` + `relation_divergence`) measure whether the
   fingerprint survived, and the measured numbers say it does.

---

## 10. Where to look in the repo

| piece | file | symbol |
|---|---|---|
| the greedy selector | `synthesize/utils.py` | `_select_relmatch`, dispatched in `_select_variant` |
| soft-labels + target wiring | `synthesize/utils.py` | inside `selector(...)` |
| CLI flag + path naming | `argument.py` | `--select-method relmatch`, `--relmatch-diag-weight` |
| build `R`, compare `R` | `validation/diagnostics.py` | `class_relation_matrix`, `relation_divergence` |
| run the diagnostic | `tools/diagnose_geometry.py` | `--select-method relmatch` |
| unit tests (worth reading!) | `tests/test_select.py` | tiny, exact, hand-checkable |
| the other selectors, for contrast | `docs/SELECTORS.md` | `covmatch`, `momentmatch`, `qddpp` |

Run it yourself:

```bash
# 1) build a relmatch distilled set (off-diagonal default)
python main.py --subset cifar100 --arch-name conv3 --ipc 10 \
               --factor 1 --mipc 300 --num-crop 5 --select-method relmatch

# 2) measure how well it preserved R (vs the real data, and compare to --select-method stock)
python tools/diagnose_geometry.py --subset cifar100 --arch-name conv3 --ipc 10 \
               --select-method relmatch
```
