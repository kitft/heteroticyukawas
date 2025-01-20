import tensorflow as tf

# Global dtype variables
real_dtype = tf.float32
complex_dtype = tf.complex64

def set_double_precision(use_double):
    global real_dtype, complex_dtype
    if use_double:
        real_dtype = tf.float64
        complex_dtype = tf.complex128
    else:
        real_dtype = tf.float32
        complex_dtype = tf.complex64

    # Optional: Set TensorFlow's default float type as well
    tf.keras.backend.set_floatx('float64' if use_double else 'float32')
    tf.keras.mixed_precision.set_global_policy('float64' if use_double else 'float32')