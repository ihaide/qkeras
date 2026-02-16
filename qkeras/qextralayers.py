from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import warnings

import numpy as np
import six
import tensorflow.compat.v2 as tf
from tensorflow.keras import activations
from tensorflow.keras import constraints
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.initializers import Initializer
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Layer
from tensorflow.python.framework import smart_cond as tf_utils

from .quantizers import *
from .quantizers import _get_integer_bits
from .quantizers import get_quantizer
from tensorflow_model_optimization.python.core.sparsity.keras.prunable_layer import (
    PrunableLayer,
)


class QNearestNeighbourManhattan(Layer):
    """
    Quantized implementation of the k nearest neighbour algorihtm. Euclediean or manhattan
    distances is used for calculating the metric. Quantization is additionally
    activated by setting  distance_quantizer and/or distance_activation.

    Args:
        - k: Number of neighbours to select
        - distance_quantizer: QKeras quantizer indicating the precision of the
          calculated distances.

    Shapes:
        - **input:**
          List of tensors [coordinates, features, n_vertices]
          coordinates :math:`(|\\mathcal{V}|, |\\mathcal{S}|)`
          features :math:`(|\\mathcal{V}|, |\\mathcal{F}|)`
          n_vertices :math:`(1)`
        - **output:**
          List of tensors [distances, neighbour_features]:
          distances :math:`(|\\mathcal{V}|, |\\mathcal{K-1}|)`
          neighbour_features :math:`(|\\mathcal{V}|, |\\mathcal{K-1}|, |\\mathcal{F}|)`
     -
    """

    def __init__(
        self,
        k,
        distance_quantizer=None,
        use_manhattan_distance=False,
        max_distance=None,
        **kwargs,
    ):
        self.k = k
        self.distance_quantizer = (
            get_quantizer(distance_quantizer)
            if distance_quantizer is not None
            else None
        )
        if max_distance is None:
            self.max_distance = (
                np.finfo(np.float32).max
                if self.distance_quantizer is None
                else 2 ** int(distance_quantizer.split("(")[1].split(",")[0])
            )
        else:
            self.max_distance = max_distance
        self.use_manhattan_distance = use_manhattan_distance
        super(QNearestNeighbourManhattan, self).__init__(**kwargs)

    def build(self, input_shapes):
        """
        Precalculate the raw mask. Is later used determine individual event size

        Raw Mask:
        0 1 2 . V
        1 2 3 . V
        2 3 4 . V
        . . . . V
        V V V V V

        """
        v = input_shapes[0][1]

        rows = tf.tile(tf.expand_dims(tf.range(v), axis=0), [v, 1])
        columns = tf.tile(tf.expand_dims(tf.range(v), axis=-1), [1, v])
        raw_mask = tf.maximum(rows, columns)
        self.raw_mask = tf.cast(raw_mask, dtype=tf.float32)

        super(QNearestNeighbourManhattan, self).build(input_shapes)

    def call(self, inputs):
        coordinates = inputs[0]
        features = inputs[1]
        active_vertices = inputs[2]

        # Calculate manhattan distances as |a-b|
        distance_matrix = tf.reduce_sum(
            tf.abs(
                tf.expand_dims(coordinates, axis=2)
                - tf.expand_dims(coordinates, axis=1)
            ),
            axis=-1,
        )
        # (B,V,V)

        b = tf.shape(inputs[0])[0]
        # 1. Expand static raw mask to batch size from (V,V) to (B,V,V)
        batch_raw_mask = tf.tile(tf.expand_dims(self.raw_mask, axis=0), [b, 1, 1])
        # 2. Expand actvie vertices tensor from (B,1) to (B,1,1)
        batch_active_vertices = tf.expand_dims(active_vertices, axis=-1)
        # 3. Calculate boolean mask
        mask = tf.less(batch_raw_mask, batch_active_vertices)
        # 4. Apply mask. Set all invalid distances to the largest float value on the current system.
        distance_matrix = tf.where(
            mask, distance_matrix, tf.zeros_like(distance_matrix) + self.max_distance
        )

        # Sort distances and select k smallest
        ranked_distances, ranked_indices = tf.nn.top_k(
            -distance_matrix, self.k
        )  # (B,V,K)

        # Remove self loop. The first distance value must always be equal to 0.
        ranked_indices = ranked_indices[:, :, 1:]  # (B,V,K-1)
        neighbour_distances = -ranked_distances[:, :, 1:]  # (B,V,K-1)

        # Gather all neighbours. We have to expand the index tensor, because we want to retrieve vertices features in the last dimension F
        # By batch_dims=1 we implicity perform the gather step for every batch seperately.
        neighbour_indices = tf.expand_dims(ranked_indices, axis=-1)  # (B,V,K-1,1)
        neighbour_features = tf.gather_nd(
            batch_dims=1, params=features, indices=neighbour_indices
        )  # (B,V,K-1,F)

        if self.distance_quantizer is not None:
            quantized_distances = self.distance_quantizer(neighbour_distances)
        else:
            quantized_distances = neighbour_distances

        return [quantized_distances, neighbour_features]

    def compute_output_shape(self, input_shapes):

        coordinate_shape = input_shapes[0]
        feature_shape = input_shapes[1]
        active_vertices_shape = input_shapes[2]

        assert len(coordinate_shape) == 3
        assert len(feature_shape) == 3
        assert len(active_vertices_shape) == 2

        assert coordinate_shape[0] == feature_shape[0]

        v = coordinate_shape[0]
        f = feature_shape[1]

        distance_shape = (v, self.k - 1)
        neighbour_feature_shape = (v, self.k - 1, f)

        return [distance_shape, neighbour_feature_shape]

    def get_config(self):
        config = {
            "k": self.k,
        }
        base_config = super(QNearestNeighbourManhattan, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_quantization_config(self):
        return {
            "distance_quantizer": str(self.distance_quantizer),
            "units": str(self.units),
        }

    def get_quantizers(self):
        return [self.distance_quantizer]


class QNearestNeighbourEuclidean(Layer):
    """
    Quantized implementation of the k nearest neighbour algorihtm. Euclediean or manhattan
    distances is used for calculating the metric. Quantization is additionally
    activated by setting  distance_quantizer and/or distance_activation.

    Args:
        - k: Number of neighbours to select
        - distance_quantizer: QKeras quantizer indicating the precision of the
          calculated distances.

    Shapes:
        - **input:**
          List of tensors [coordinates, features, n_vertices]
          coordinates :math:`(|\\mathcal{V}|, |\\mathcal{S}|)`
          features :math:`(|\\mathcal{V}|, |\\mathcal{F}|)`
          n_vertices :math:`(1)`
        - **output:**
          List of tensors [distances, neighbour_features]:
          distances :math:`(|\\mathcal{V}|, |\\mathcal{K-1}|)`
          neighbour_features :math:`(|\\mathcal{V}|, |\\mathcal{K-1}|, |\\mathcal{F}|)`
     -
    """

    def __init__(
        self,
        k,
        distance_quantizer=None,
        use_manhattan_distance=False,
        max_distance=None,
        **kwargs,
    ):
        self.k = k
        self.distance_quantizer = (
            get_quantizer(distance_quantizer)
            if distance_quantizer is not None
            else None
        )
        if max_distance is None:
            self.max_distance = (
                np.finfo(np.float32).max
                if self.distance_quantizer is None
                else 2 ** int(distance_quantizer.split("(")[1].split(",")[0])
            )
        else:
            self.max_distance = max_distance
        self.use_manhattan_distance = use_manhattan_distance
        super(QNearestNeighbourEuclidean, self).__init__(**kwargs)

    def build(self, input_shapes):
        """
        Precalculate the raw mask. Is later used determine individual event size

        Raw Mask:
        0 1 2 . V
        1 2 3 . V
        2 3 4 . V
        . . . . V
        V V V V V

        """
        v = input_shapes[0][1]

        rows = tf.tile(tf.expand_dims(tf.range(v), axis=0), [v, 1])
        columns = tf.tile(tf.expand_dims(tf.range(v), axis=-1), [1, v])
        raw_mask = tf.maximum(rows, columns)
        self.raw_mask = tf.cast(raw_mask, dtype=tf.float32)

        super(QNearestNeighbourEuclidean, self).build(input_shapes)

    def call(self, inputs):
        coordinates = inputs[0]
        features = inputs[1]
        active_vertices = inputs[2]

        # Calculate squared euclidean distances as (a-b)^2 = a^2 + b^2 - 2ab
        sub_factor = -2 * tf.matmul(
            coordinates, tf.transpose(coordinates, perm=[0, 2, 1])
        )  # -2ab term
        dotA = tf.expand_dims(
            tf.reduce_sum(coordinates * coordinates, axis=2), axis=2
        )  # a^2 term
        dotB = tf.expand_dims(
            tf.reduce_sum(coordinates * coordinates, axis=2), axis=1
        )  # b^2 term
        distance_matrix = tf.abs(sub_factor + dotA + dotB)  # (B,V,V)

        b = tf.shape(inputs[0])[0]
        # 1. Expand static raw mask to batch size from (V,V) to (B,V,V)
        batch_raw_mask = tf.tile(tf.expand_dims(self.raw_mask, axis=0), [b, 1, 1])
        # 2. Expand actvie vertices tensor from (B,1) to (B,1,1)
        batch_active_vertices = tf.expand_dims(active_vertices, axis=-1)
        # 3. Calculate boolean mask
        mask = tf.less(batch_raw_mask, batch_active_vertices)
        # 4. Apply mask. Set all invalid distances to the largest float value on the current system.
        distance_matrix = tf.where(
            mask, distance_matrix, tf.zeros_like(distance_matrix) + self.max_distance
        )

        # Sort distances and select k smallest
        ranked_distances, ranked_indices = tf.nn.top_k(
            -distance_matrix, self.k
        )  # (B,V,K)

        # Remove self loop. The first distance value must always be equal to 0.
        ranked_indices = ranked_indices[:, :, 1:]  # (B,V,K-1)
        neighbour_distances = -ranked_distances[:, :, 1:]  # (B,V,K-1)

        # Gather all neighbours. We have to expand the index tensor, because we want to retrieve vertices features in the last dimension F
        # By batch_dims=1 we implicity perform the gather step for every batch seperately.
        neighbour_indices = tf.expand_dims(ranked_indices, axis=-1)  # (B,V,K-1,1)
        neighbour_features = tf.gather_nd(
            batch_dims=1, params=features, indices=neighbour_indices
        )  # (B,V,K-1,F)

        if self.distance_quantizer is not None:
            quantized_distances = self.distance_quantizer(neighbour_distances)
        else:
            quantized_distances = neighbour_distances

        return [quantized_distances, neighbour_features]

    def compute_output_shape(self, input_shapes):

        coordinate_shape = input_shapes[0]
        feature_shape = input_shapes[1]
        active_vertices_shape = input_shapes[2]

        assert len(coordinate_shape) == 3
        assert len(feature_shape) == 3
        assert len(active_vertices_shape) == 2

        assert coordinate_shape[0] == feature_shape[0]

        v = coordinate_shape[0]
        f = feature_shape[1]

        distance_shape = (v, self.k - 1)
        neighbour_feature_shape = (v, self.k - 1, f)

        return [distance_shape, neighbour_feature_shape]

    def get_config(self):
        config = {
            "k": self.k,
        }
        base_config = super(QNearestNeighbourEuclidean, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_quantization_config(self):
        return {
            "distance_quantizer": str(self.distance_quantizer),
            "units": str(self.units),
        }

    def get_quantizers(self):
        return [self.distance_quantizer]


class QGlobalAverageReduce1D(Layer):
    """
    This Layer Reduces the innermost dimension of an arbitrary input tensor
    by calculating the arithmetic average on it.
    Args:
     - kernel_quantizer: Quantization of the output value

     - divider_quantizer: Quantization of the divider value
    Shapes:
     - **:input:**
       Tensor of rank > 0
       n: Number of active inputs. Muste be a tensor of rank 0.
     - **:output:**
       Input tensor rank-1
    """

    def __init__(self, kernel_quantizer=None, divider_quantizer=None, **kwargs):
        if kernel_quantizer is not None:
            self.kernel_quantizer = get_quantizer(kernel_quantizer)
        else:
            self.kernel_quantizer = None
        if divider_quantizer is not None:
            self.divider_quantizer = get_quantizer(divider_quantizer)
        else:
            self.divider_quantizer = None
        super(QGlobalAverageReduce1D, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) > 1
        self.n = tf.cast(input_shape[-1], tf.float32)
        super(QGlobalAverageReduce1D, self).build(input_shape)

    def call(self, input):
        # Separation of addition and division to account for loss in precision

        if self.divider_quantizer is not None:
            n_inv = self.divider_quantizer(tf.cast(1., tf.float32) / self.n)
        else:
            n_inv = 1. / self.n

        sum = tf.math.reduce_sum(input, axis=-1)
        output = tf.math.multiply(sum, n_inv)

        if self.kernel_quantizer is not None:
            quantized_output = self.kernel_quantizer(output)
        else:
            quantized_output = output

        return quantized_output

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

    def get_config(self):
        return super(QGlobalAverageReduce1D, self).get_config()

    def get_quantization_config(self):
        return {
            "kernel_quantizer": str(self.kernel_quantizer),
            "divider_quantizer": str(self.kernel_quantizer),
            "units": str(self.units),
        }

    def get_quantizers(self):
        return [self.kernel_quantizer, self.divider_quantizer]


class GlobalMaxReduce1D(Layer):
    """
    This Layer Reduces the innermost dimension of an arbitrary input tensor
    by calculating the Max() funcion on it.
    Args:
     - None
    Shapes:
     - **:input:**
       Tensor of rank
     - **:output:**
       Input tensor rank-1
    """

    def __init__(self, **kwargs):
        super(GlobalMaxReduce1D, self).__init__(**kwargs)

    def call(self, inputs):
        return K.max(inputs, axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

    def get_config(self):
        return super(GlobalMaxReduce1D, self).get_config()


class QExponential(Layer):
    """
    Implements a quantized element-wise exponential function.
    Formula is given by :math:`y = \exp(\alpha x)`
    Args:
        - alpha: constant in formula as shown above
        - exponential_quantizer: Quantizer confgiuration. Should be of type
          quantized_exp(...)
    """

    def __init__(self, alpha=1.0, exponential_quantizer=None, **kwargs):
        self.alpha = alpha
        if exponential_quantizer is not None:
            self.exponential_quantizer = get_quantizer(exponential_quantizer)
        else:
            self.exponential_quantizer = None
        super(QExponential, self).__init__(**kwargs)

    def call(self, input):

        x = tf.multiply(self.alpha, input)

        if self.exponential_quantizer is not None:
            output = self.exponential_quantizer(x)
        else:
            output = tf.keras.backend.exp(x)

        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "alpha": self.alpha,
        }
        base_config = super(QExponential, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_quantization_config(self):
        return {
            "exponential_quantizer": str(self.exponential_quantizer),
            "units": str(self.units),
        }

    def get_quantizers(self):
        return [self.exponential_quantizer]


class QDistanceWeighting(Layer):
    """
    Implements a (not yet) quantized element-wise weighting function.
    Used to apply a distance-based weighting in GravNet.
    Different weighting functions are available:
    - exponential: :math:`y = \exp(\alpha x)` (default, same as QExponential)
    - quadratic: :math:`y = \alpha*(x - \beta)^2 + \epsilon` if x > \beta,
      else :math:`y = \epsilon` where \beta is chosen such that f(x=0) = 1:
      :math:`\beta = \sqrt{(1 - \epsilon) / \alpha}`
    - linear: :math:`y = 1 - \alpha * x` if x < 1/\alpha,
      else :math:`y = \epsilon`

    Args:
        - alpha: constant in formula as shown above
        - epsilon: small constant to avoid returning zero
        - exponential_quantizer: Quantizer configuration. Should be of type
          quantized_exp(...) or None
        - quadratic_quantizer: Quantizer configuration for quadratic weighting.
          Currently not implemented, any non-None value will use the
          quadratic weighting function.
        - linear_quantizer: Quantizer configuration for linear weighting.
          Currently not implemented, any non-None value will use the
          linear weighting function.
        - input_dtype: Data type of the input tensor, default is tf.float32
        - function: Not used, just for showing the type in the config.
    """

    def __init__(
            self,
            alpha=1.0,
            epsilon=0.01,
            max_value=1.0,
            exponential_quantizer=None,
            quadratic_quantizer=None,
            linear_quantizer=None,
            input_dtype=tf.float32,
            function=None,
            **kwargs
        ):
        self.alpha = tf.convert_to_tensor(alpha, dtype=input_dtype)
        self.epsilon = tf.convert_to_tensor(epsilon, dtype=input_dtype)
        self.max_value = tf.convert_to_tensor(max_value, dtype=input_dtype)
        self.beta = K.sqrt((self.max_value - self.epsilon) / self.alpha)
        if exponential_quantizer is not None:
            self.exponential_quantizer = get_quantizer(exponential_quantizer)
            self.function = "exponential"
        else:
            self.exponential_quantizer = None
            self.function = "exponential_no_q"  # Default function is exponential
        # TODO: Implement quadratic and linear quantizers more elegantly
        if quadratic_quantizer is not None:
            self.quadratic_quantizer = quadratic_quantizer
            self.function = "quadratic"
        else:
            self.quadratic_quantizer = None
        if linear_quantizer is not None:
            self.linear_quantizer = linear_quantizer
            self.function = "linear"
        else:
            self.linear_quantizer = None
        super(QDistanceWeighting, self).__init__(**kwargs)

    def call(self, input):

        if self.exponential_quantizer is not None:
            bits = self.exponential_quantizer.bits
            input_quantizer = get_quantizer(f"quantized_linear({bits+2}, 0)")
            x = input_quantizer(input)
            x = tf.multiply(self.alpha, x)
            output = self.exponential_quantizer(x)
            output = tf.where(input >= 0.5-2**(-(bits+1)), 0.0, output)
        elif self.quadratic_quantizer is not None:
            x = tf.subtract(input, self.beta)
            x = tf.square(x)
            x = tf.multiply(self.alpha, x)
            # Use tf.where to select epsilon when input < beta, else quadratic value + epsilon
            output = tf.where(
                tf.less(self.beta, input),
                self.epsilon,
                x + self.epsilon
            )
        elif self.linear_quantizer is not None:
            x = tf.multiply(self.alpha, input)
            x = tf.subtract(self.max_value, x)
            x = K.clip(x, self.epsilon, None)
            output = get_quantizer(self.linear_quantizer)(x)
        else:
            x = tf.multiply(self.alpha, input)
            output = tf.keras.backend.exp(x)

        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "alpha": self.alpha.numpy().item(),
            "epsilon": self.epsilon.numpy().item(),
            "max_value": self.max_value.numpy().item(),
            "function": self.function,
        }
        base_config = super(QDistanceWeighting, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_quantization_config(self):
        if self.exponential_quantizer is not None:
            return {
                "exponential_quantizer": str(self.exponential_quantizer),
                "units": str(self.units),
            }
        if self.quadratic_quantizer is not None:
            return {
                "quadratic_quantizer": str(self.quadratic_quantizer),
                "units": str(self.units),
            }
        if self.linear_quantizer is not None:
            return {
                "linear_quantizer": str(self.linear_quantizer),
                "units": str(self.units),
            }
        return {
            "exponential_quantizer": str(self.exponential_quantizer),
            "units": str(self.units),
        }

    def get_quantizers(self):
        if self.exponential_quantizer is not None:
            return [self.exponential_quantizer]
        if self.quadratic_quantizer is not None:
            return [self.quadratic_quantizer]
        if self.linear_quantizer is not None:
            return [self.linear_quantizer]
        # Default to exponential quantizer if none specified
        return [self.exponential_quantizer]
