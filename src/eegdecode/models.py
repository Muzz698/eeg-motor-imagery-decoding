"""The decoders we compare.

1. CSP + LDA: the classic motor-imagery baseline. Common Spatial Patterns finds
   channel combinations whose power differs most between the two classes; a
   linear discriminant classifies the log-power of those components.
2. EEGNet (Lawhern et al., 2018): a compact convolutional network designed for
   EEG. It learns temporal filters, then spatial filters, end to end.
3. C3/C4 + LDA: a one-feature baseline with no learned spatial filters. A linear
   discriminant classifies the log-power difference between electrodes C3 and C4.
"""
from __future__ import annotations

import os

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline


def make_csp_lda(n_components: int = 6, reg: str | float | None = "ledoit_wolf") -> Pipeline:
    from mne.decoding import CSP

    return Pipeline([
        ("csp", CSP(n_components=n_components, reg=reg, log=True, norm_trace=False)),
        ("lda", LinearDiscriminantAnalysis()),
    ])


class LogPowerDifference(BaseEstimator, TransformerMixin):
    """One feature per trial: log-power of channel `left` minus channel `right`."""

    def __init__(self, left: int = 8, right: int = 12):
        self.left = left
        self.right = right

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        diff = np.log(X[:, self.left].var(axis=-1)) - np.log(X[:, self.right].var(axis=-1))
        return diff[:, np.newaxis]


def make_c3c4_lda(c3_index: int = 8, c4_index: int = 12) -> Pipeline:
    """Imagining one hand lowers mu/beta power over the opposite hemisphere, so this
    needs no spatial filters. The default indices are C3 and C4 in PhysioNet's order."""
    return Pipeline([
        ("lateralization", LogPowerDifference(c3_index, c4_index)),
        ("lda", LinearDiscriminantAnalysis()),
    ])


def _keras():
    """Import Keras 3 with the PyTorch backend by default.

    PyTorch is more stable than TensorFlow on macOS and in conda environments,
    where TensorFlow can crash with a segmentation fault. To use TensorFlow
    instead, set the environment variable KERAS_BACKEND=tensorflow.

    Training runs on the CPU by default. EEGNet is small, so the CPU is fast
    enough, and Apple's GPU backend (MPS) gave wrong results in testing. To use
    an NVIDIA GPU (for example on Colab), set KERAS_TORCH_DEVICE=cuda.
    """
    os.environ.setdefault("KERAS_BACKEND", "torch")
    os.environ.setdefault("KERAS_TORCH_DEVICE", "cpu")
    import keras

    return keras


def build_eegnet(n_channels: int, n_samples: int, n_classes: int = 2, F1: int = 8,
                 D: int = 2, F2: int = 16, kern_length: int = 64, dropout: float = 0.5,
                 learning_rate: float = 1e-3):
    """EEGNet-8,2 as described in Lawhern et al. (2018), J. Neural Eng."""
    keras = _keras()
    layers = keras.layers

    inputs = keras.Input(shape=(n_channels, n_samples, 1))
    # Block 1: temporal convolution (acts like a bank of band-pass filters) ...
    x = layers.Conv2D(F1, (1, kern_length), padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    # ... then a depthwise spatial filter per temporal filter (like learned CSP).
    x = layers.DepthwiseConv2D((n_channels, 1), use_bias=False, depth_multiplier=D,
                               depthwise_constraint=keras.constraints.MaxNorm(1.0))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("elu")(x)
    x = layers.AveragePooling2D((1, 4))(x)
    x = layers.Dropout(dropout)(x)
    # Block 2: separable convolution mixes feature maps over time.
    x = layers.SeparableConv2D(F2, (1, 16), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("elu")(x)
    x = layers.AveragePooling2D((1, 8))(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(n_classes, activation="softmax",
                           kernel_constraint=keras.constraints.MaxNorm(0.25))(x)

    model = keras.Model(inputs, outputs, name="EEGNet")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def _scale(X: np.ndarray, scale: float) -> np.ndarray:
    """Remove each channel's offset, then divide ALL channels by one number.

    Motor imagery shows up as a difference in power BETWEEN channels (for example
    C3 vs C4). Scaling every channel to unit variance separately would erase that
    difference, so we use a single scale factor computed on the training data
    only (no test information leaks in).
    """
    X = X - X.mean(axis=-1, keepdims=True)
    return (X / scale).astype(np.float32)


class EEGNetClassifier(BaseEstimator, ClassifierMixin):
    """scikit-learn style wrapper so EEGNet runs through the same evaluation code.

    Trains for a fixed number of epochs, with no early stopping. On real EEG the
    validation loss on new people is lowest before the model has learned anything,
    because its confidence grows faster than its accuracy. Early stopping on that
    loss restored a near-untrained model and left EEGNet at chance.
    """

    def __init__(self, epochs: int = 100, batch_size: int = 64, F1: int = 8, D: int = 2,
                 F2: int = 16, kern_length: int = 64, dropout: float = 0.5,
                 learning_rate: float = 1e-3, seed: int = 42, verbose: int = 0):
        self.epochs = epochs
        self.batch_size = batch_size
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.kern_length = kern_length
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.seed = seed
        self.verbose = verbose

    def fit(self, X, y):
        keras = _keras()
        keras.utils.set_random_seed(self.seed)
        self.classes_ = np.unique(y)
        self.scale_ = float(X.std()) + 1e-6
        X_tr = _scale(X, self.scale_)[..., np.newaxis]

        self.model_ = build_eegnet(X_tr.shape[1], X_tr.shape[2], len(self.classes_), self.F1,
                                   self.D, self.F2, self.kern_length, self.dropout,
                                   self.learning_rate)
        self.history_ = self.model_.fit(X_tr, y, epochs=self.epochs, batch_size=self.batch_size,
                                        verbose=self.verbose).history
        print(f"      EEGNet: {self.epochs} epochs, final training accuracy "
              f"{self.history_['accuracy'][-1]:.3f}", flush=True)
        return self

    def predict_proba(self, X):
        return self.model_.predict(_scale(X, self.scale_)[..., np.newaxis], verbose=0)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
