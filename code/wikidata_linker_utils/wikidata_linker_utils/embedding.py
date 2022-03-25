import tensorflow as tf
import math

EMBEDDING_CPU_DEVICE = "EMBEDDING_CPU_DEVICE"
EMBEDDING_VARIABLES = "EMBEDDING_VARIABLES"


def embed_latlong(inputs, embed_dimension, dimension, is_training):
    mask = tf.cast(inputs[:, 3], tf.bool)
    with_latlong = inputs[mask]
    h = tf.contrib.layers.fully_connected(with_latlong, embed_dimension, activation_fn=tf.nn.relu,
                                          scope="Embed")
    sparse_res = tf.contrib.layers.fully_connected(h, dimension, activation_fn=tf.tanh,
                                                   scope="Proj")
    # missing latlong info
    blank = tf.get_variable(name="Blank",
                            trainable=True,
                            shape=[1, dimension])
    scatter_indices = tf.where(mask)
    dense_res = tf.scatter_nd(scatter_indices, sparse_res, shape=(tf.shape(inputs)[0], dimension))
    return tf.where(mask, dense_res, tf.tile(blank, [tf.shape(inputs)[0], 1]), name="LatLongEmbedding")



def embed_date(inputs, dimension, embed_ranges, is_training):
    # 1 if present, 0 otherwise
    date_present = tf.cast(inputs[:, 1], tf.bool)
    dates = inputs[:, 0]
    all_embedded = []
    for idx, date_range in enumerate(embed_ranges):
        with tf.variable_scope(f"DateEmbed{idx}"):
            start = date_range["start"]
            stop = date_range["stop"]
            max_element = math.ceil((stop - start) / date_range["divisor"]) + 1
            W = get_embedding_lookup(max_element, dimension, tf.float32,
                                     initializer=tf.random_uniform_initializer(
                -1.0 / (len(embed_ranges) * math.sqrt(dimension)),
                1.0 / (len(embed_ranges) * math.sqrt(dimension))
            ))
            zeroed_date = dates - start
            if date_range["divisor"] != 1:
                zeroed_date = zeroed_date // date_range["divisor"]
            embedded = tf.nn.embedding_lookup(W, tf.clip_by_value(zeroed_date, 0, max_element - 1),
                                              name="DateEmbed{}".format(date_range["divisor"]))
            all_embedded.append(embedded)
    all_embedded = tf.add_n(all_embedded)
    blank = tf.get_variable(name="Blank",
                            trainable=True,
                            shape=[1, dimension])
    return tf.where(date_present, all_embedded, tf.tile(blank, [tf.shape(inputs)[0], 1]), name="DateEmbedding")


def get_embedding_lookup(size, dim, dtype, reuse=None, trainable=True,
                         place_on_cpu_if_big=False, initializer=None):
    if initializer is None:
        initializer = tf.random_uniform_initializer(
            -1.0 / math.sqrt(dim),
            1.0 / math.sqrt(dim)
        )
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
        if place_on_cpu_if_big and size > 100000:
            with tf.device("cpu"):
                W = tf.get_variable(
                    name="embedding",
                    shape=[size, dim],
                    dtype=dtype,
                    initializer=initializer,
                    trainable=trainable)
                tf.add_to_collection(EMBEDDING_CPU_DEVICE, W)
        else:
            W = tf.get_variable(
                name="embedding",
                shape=[size, dim],
                dtype=dtype,
                initializer=initializer,
                trainable=trainable)
        return W


def embedding_lookup(inputs,
                     size,
                     dim,
                     dtype,
                     reuse=None,
                     mask_negative=False,
                     trainable=True,
                     place_on_cpu_if_big=True,
                     name=None):
    """
    Construct an Embedding layer that gathers
    elements from a matrix with `size` rows,
    and `dim` features using the indices stored in `x`.

    Arguments:
    ----------
        inputs : tf.Tensor, of integer type
        size : int, how many symbols in the lookup table
        dim : int, how many columns per symbol.
        dtype : data type for the lookup table (e.g. tf.float32)
        reuse : bool, (default None) whether the lookup table
            was already used before (thus this is weight sharing).
        mask_negative : bool, (default False) should -1s in the
            lookup input indicate padding (e.g. no lookup),
            and thus should those values be masked out post-lookup.
        trainable : bool (default True), whether the parameters of
            this lookup table can be backpropagated into (e.g.
            for Glove word vectors that are fixed pre-trained, this
            can be set to False).
        place_on_cpu_if_big : bool, if matrix is big, store it on cpu.
    Returns:
    --------
        tf.Tensor, result of tf.nn.embedding_lookup(LookupTable, inputs)
    """
    W = get_embedding_lookup(size, dim, dtype, reuse, trainable=trainable,
                             place_on_cpu_if_big=place_on_cpu_if_big)
    tf.add_to_collection(EMBEDDING_VARIABLES, W)
    if mask_negative:
        embedded = tf.nn.embedding_lookup(W, tf.maximum(inputs, 0))
        null_mask = tf.expand_dims(tf.cast(tf.not_equal(inputs, -1), dtype), -1)
        return W, embedded * null_mask
    else:
        return W, tf.nn.embedding_lookup(W, inputs)
