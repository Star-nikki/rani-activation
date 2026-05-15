"""
rani/tf_rani.py
===============
RANI: Rational Adaptive Normalizing Initialization
TensorFlow / Keras Implementation

Formula
-------
    RANI(x) = lambda * x * (1 + alpha * |x|) / (1 + |x|)   for x >= 0
    RANI(x) = lambda * alpha * (exp(x) - 1)                  for x < 0

See rani/torch_rani.py for the full mathematical description and
derivation of the initialization values alpha=1.6733, lam=1.0507.

Author : Nikki Rani
Paper  : doi.org/10.5281/zenodo.XXXXXXX
PyPI   : pip install rani-activation
GitHub : github.com/Star-nikki/rani-activation
"""

import math
import tensorflow as tf


# =============================================================================
# FUNCTIONAL API
# Use this when you want fixed (non-learnable) alpha and lam values.
# =============================================================================

def rani(x, alpha: float = 1.6733, lam: float = 1.0507):
    """
    Apply the RANI activation function with fixed parameters.

    This is the functional (non-learnable) version. For learnable parameters
    inside a Keras model, use the RANI Layer class below instead.

    Args:
        x     : Input tensor of any shape and numeric dtype.
        alpha : Asymmetry parameter. Default 1.6733.
        lam   : Scale parameter. Default 1.0507.

    Returns:
        Tensor of same shape as x with RANI applied element-wise.

    Example:
        >>> import tensorflow as tf
        >>> from rani.tf_rani import rani
        >>> x = tf.constant([-3.0, -1.0, 0.0, 1.0, 3.0])
        >>> rani(x)
    """
    x   = tf.cast(x, tf.float32)
    pos = lam * x * (1.0 + alpha * tf.abs(x)) / (1.0 + tf.abs(x))
    neg = lam * alpha * (tf.exp(x) - 1.0)
    return tf.where(x >= 0, pos, neg)


# =============================================================================
# KERAS LAYER API
# Use this inside tf.keras.Sequential or any Keras model.
# alpha and lam are learned automatically by the optimizer.
# Fully serializable — model.save() and model.load() work correctly.
# =============================================================================

class RANI(tf.keras.layers.Layer):
    """
    RANI: Rational Adaptive Normalizing Initialization — Keras Layer.

    Self-normalizing activation function with learnable per-layer parameters.
    Both alpha and lam are updated by the optimizer during training and are
    fully serializable for model.save() and tf.keras.models.load_model().

    Args:
        alpha (float): Initial asymmetry parameter. Default 1.6733.
        lam   (float): Initial scale parameter.    Default 1.0507.

    Call arguments:
        x: Input tensor of any shape and numeric dtype.

    Output:
        Tensor of same shape and dtype as input.

    Examples:
        Inside a Sequential model::

            >>> import tensorflow as tf
            >>> from rani.tf_rani import RANI
            >>>
            >>> model = tf.keras.Sequential([
            ...     tf.keras.layers.Dense(512),
            ...     RANI(),
            ...     tf.keras.layers.Dense(256),
            ...     RANI(),
            ...     tf.keras.layers.Dense(10, activation='softmax'),
            ... ])
            >>> model.build(input_shape=(None, 784))
            >>> model.summary()

        Saving and loading::

            >>> model.save('my_model.keras')
            >>> loaded = tf.keras.models.load_model(
            ...     'my_model.keras',
            ...     custom_objects={'RANI': RANI}
            ... )

        Checking learned parameter values::

            >>> act = RANI()
            >>> act.build(())
            >>> print(f"alpha = {act.alpha.numpy():.4f}")
            alpha = 1.6733
            >>> # ... after training ...
            >>> print(f"alpha = {act.alpha.numpy():.4f}")
            alpha = 1.5821   # learned a different value

    Reference:
        Rani, N. (2025). RANI: Rational Adaptive Normalizing Initialization.
        Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX
    """

    def __init__(self,
                 alpha: float = 1.6733,
                 lam:   float = 1.0507,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        # Store init values — used in build() to create the weights
        self._alpha_init = float(alpha)
        self._lam_init   = float(lam)

    def build(self, input_shape) -> None:
        """
        Create the trainable weight variables alpha and lam.
        Called automatically the first time you call the layer.
        """
        self.alpha = self.add_weight(
            name        = "alpha",
            shape       = (),              # scalar — one value per layer
            dtype       = tf.float32,
            initializer = tf.constant_initializer(self._alpha_init),
            trainable   = True,
        )
        self.lam = self.add_weight(
            name        = "lam",
            shape       = (),              # scalar — one value per layer
            dtype       = tf.float32,
            initializer = tf.constant_initializer(self._lam_init),
            trainable   = True,
        )
        super().build(input_shape)

    def call(self, x):
        """
        Apply RANI activation element-wise.

        Args:
            x: Input tensor of any shape.

        Returns:
            Tensor of same shape as input.
        """
        x   = tf.cast(x, tf.float32)

        # Positive branch: rational — no transcendental operations
        pos = self.lam * x * (1.0 + self.alpha * tf.abs(x)) / (1.0 + tf.abs(x))

        # Negative branch: exponential — same as SELU's negative branch
        neg = self.lam * self.alpha * (tf.exp(x) - 1.0)

        # Combine: use positive branch where x >= 0, negative branch where x < 0
        return tf.where(x >= 0, pos, neg)

    def get_config(self) -> dict:
        """
        Return layer configuration as a dictionary.

        Required for model.save() to work. Stores the current learned
        values of alpha and lam so they are preserved when saving.
        """
        base_config = super().get_config()
        base_config.update({
            "alpha": float(self.alpha.numpy()),
            "lam":   float(self.lam.numpy()),
        })
        return base_config

    @classmethod
    def from_config(cls, config: dict) -> "RANI":
        """
        Reconstruct a RANI layer from its config dictionary.

        Required for model.load() to work correctly.
        Called automatically by tf.keras.models.load_model().
        """
        return cls(**config)


# =============================================================================
# WEIGHT INITIALIZATION
# =============================================================================

class LeCunNormal(tf.keras.initializers.Initializer):
    """
    LeCun normal initializer — mathematically correct init for RANI.

    Samples weights from N(0, 1/fan_in). Derived from the variance
    preservation condition in the self-normalizing fixed-point analysis.

    Use this when building Dense layers in a RANI network:

        layer = tf.keras.layers.Dense(
            128,
            kernel_initializer=LeCunNormal()
        )

    Note: tf.keras.initializers.LecunNormal() does the same thing.
    This class is defined explicitly so the derivation is transparent.
    """

    def __call__(self, shape, dtype=tf.float32):
        fan_in = shape[-2] if len(shape) >= 2 else shape[0]
        std    = 1.0 / math.sqrt(float(fan_in))
        return tf.random.normal(shape, mean=0.0, stddev=std, dtype=dtype)

    def get_config(self) -> dict:
        return {}


def build_rani_model(layer_sizes: list,
                     output_size: int,
                     input_size:  int) -> tf.keras.Model:
    """
    Build a Keras Sequential model with RANI activations and LeCun init.

    Convenience function for quickly building RANI networks.

    Args:
        layer_sizes  : List of hidden layer sizes. E.g. [512, 256, 128].
        output_size  : Number of output neurons (number of classes).
        input_size   : Number of input features.

    Returns:
        Compiled tf.keras.Sequential model.

    Example:
        >>> from rani.tf_rani import build_rani_model
        >>> model = build_rani_model(
        ...     layer_sizes=[512, 256, 128],
        ...     output_size=10,
        ...     input_size=784
        ... )
        >>> model.summary()
    """
    lecun = LeCunNormal()
    layers = [tf.keras.layers.InputLayer(input_shape=(input_size,))]

    for size in layer_sizes:
        layers.append(
            tf.keras.layers.Dense(
                size,
                kernel_initializer=lecun,
                bias_initializer='zeros',
                use_bias=True,
            )
        )
        layers.append(tf.keras.layers.BatchNormalization())
        layers.append(RANI())
        layers.append(tf.keras.layers.Dropout(0.2))

    layers.append(
        tf.keras.layers.Dense(
            output_size,
            kernel_initializer=lecun,
            activation='softmax',
        )
    )

    model = tf.keras.Sequential(layers)
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss      = 'sparse_categorical_crossentropy',
        metrics   = ['accuracy'],
    )
    return model


# =============================================================================
# QUICK TEST — run this file directly to verify the implementation
# python rani/tf_rani.py
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("RANI TensorFlow/Keras Implementation — Quick Verification")
    print("=" * 60)

    # Build layer and force it to create weights
    act = RANI()
    act.build(())

    print(f"\nalpha : {act.alpha.numpy():.4f}  (initialized at fixed-point value)")
    print(f"lam   : {act.lam.numpy():.4f}  (initialized at fixed-point value)")

    # Test key values
    print("\nKey value checks:")
    print(f"  RANI(0.0)   = {act(tf.constant([0.0])).numpy()[0]:.6f}   (expected: 0.0)")
    print(f"  RANI(-100)  = {act(tf.constant([-100.0])).numpy()[0]:.6f}  (expected: ~-1.7582)")
    print(f"  RANI(1.0)   = {act(tf.constant([1.0])).numpy()[0]:.6f}   (expected: > 0)")
    print(f"  RANI(-1.0)  = {act(tf.constant([-1.0])).numpy()[0]:.6f}  (expected: < 0)")
    print(f"  RANI(-500)  = {act(tf.constant([-500.0])).numpy()[0]:.6f}  (expected: finite)")
    print(f"  RANI(500)   = {act(tf.constant([500.0])).numpy()[0]:.6f}   (expected: finite)")

    # Shape test
    x_batch = tf.random.normal((32, 128))
    y_batch = act(x_batch)
    print(f"\nShape test: input {x_batch.shape} → output {y_batch.shape}  (expected: same)")

    # Serialization test
    config    = act.get_config()
    restored  = RANI.from_config(config)
    restored.build(())
    print(f"\nSerialization test:")
    print(f"  Original alpha : {act.alpha.numpy():.4f}")
    print(f"  Restored alpha : {restored.alpha.numpy():.4f}  (expected: same)")

    # Test inside Sequential model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, input_shape=(32,)),
        RANI(),
        tf.keras.layers.Dense(10),
    ])
    x   = tf.random.normal((8, 32))
    out = model(x)
    print(f"\nSequential model test: output shape {out.shape}  (expected: (8, 10))")

    # Learnable test
    optim = tf.keras.optimizers.Adam(learning_rate=1e-2)
    alpha_before = float(act.alpha.numpy())
    lam_before   = float(act.lam.numpy())
    x = tf.random.normal((32, 64))
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean(act(x))
    grads = tape.gradient(loss, act.trainable_variables)
    optim.apply_gradients(zip(grads, act.trainable_variables))
    print(f"\nAfter one optimizer step:")
    print(f"  alpha: {act.alpha.numpy():.6f}  (should differ from {alpha_before})")
    print(f"  lam:   {act.lam.numpy():.6f}  (should differ from {lam_before})")

    # NaN check
    import numpy as np
    extreme = tf.constant([-500., -100., 0., 100., 500.])
    out_ext = act(extreme).numpy()
    nan_check = np.isnan(out_ext).any() or np.isinf(out_ext).any()
    print(f"\nNaN/Inf check on extreme inputs: {'FAILED' if nan_check else 'PASSED'}")

    print("\n" + "=" * 60)
    print("All checks done. Run pytest tests/test_tf.py for full tests.")
    print("=" * 60)