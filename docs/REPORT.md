# Decoding imagined hand movements from EEG: does deep learning generalize across people?

**Author:** Muzamil · **Code:** github.com/Muzz698/eeg-motor-imagery-decoding

> Fill in each section after running the experiments. Aim for 2–4 pages in total, then export to PDF. Keep it honest: a clear negative result is still a good result.

## 1. Question

Brain-computer interfaces usually need a calibration session for each new user. I ask (1) how well a classical decoder (CSP + LDA) and a deep network (EEGNet) decode left- versus right-hand motor imagery, and (2) how much accuracy is lost when the decoder must work on people it was never trained on.

_Add one or two sentences on why this matters (for example, calibration time for patients) and what you expected._

## 2. Data and preprocessing

_Number of subjects and trials actually used. Filter band, epoch window, resampling. Why 7–30 Hz (mu and beta rhythms, event-related desynchronization)._

## 3. Methods

_CSP + LDA in two sentences. EEGNet in three sentences. The two evaluation schemes, and why the cross-subject split is done by subject rather than by trial. The chance-level calculation._

## 4. Results

_Table of mean ± std accuracy. The per-subject figure. Number of subjects above chance. The size of the drop from within- to cross-subject._

## 5. Discussion

_Which method won and why you think so. Variation between people: some subjects are near chance for every method ("BCI illiteracy"). What the spatial patterns or learned filters show, if you plotted them (for example, CSP patterns over C3/C4)._

## 6. Limitations and next steps

_Small number of trials, single band, no artifact removal. What you would try next: transfer learning with a few calibration trials, Riemannian methods, more subjects._

## References

_Copy from the README._
