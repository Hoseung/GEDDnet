import tensorflow.compat.v1 as tf
from GEDDnet import GEDDnet_infer
import numpy as np

tf.disable_v2_behavior()  # To use TensorFlow 1.x functions in TensorFlow 2 environment

input_size = (64, 96)

mu = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape(
    (1, 1, 3))


x_f = tf.placeholder(tf.float32, [None, input_size[1], input_size[1], 3])
x_l = tf.placeholder(tf.float32, [None, input_size[0], input_size[1], 3])
x_r = tf.placeholder(tf.float32, [None, input_size[0], input_size[1], 3])

y_conv, face_h_trans, h_trans = GEDDnet_infer(x_f, x_l, x_r, mu,
                                                vgg_path='../data/vgg16_weights.npz',
                                                num_subj=50)


all_var = tf.global_variables()
var_to_restore = []
for var in all_var:
    if 'Momentum' in var.name or 'LearnRate' in var.name:
        continue
    var_to_restore.append(var)


saver = tf.train.Saver(var_to_restore)

with tf.Session() as sess:
    # Load the graph
    #saver = tf.train.import_meta_graph('../data/models/model19.ckpt.meta')

    # Restore weights
    saver.restore(sess, '../data/models/model19.ckpt')

    input_tensor = tf.get_default_graph().get_tensor_by_name("Placeholder:0")
    output_tensor = tf.get_default_graph().get_tensor_by_name("truediv_3:0")

    # Export the model to SavedModel format
    tf.saved_model.simple_save(
        session=sess,
        export_dir="../data/model/V2",
        inputs={"input": input_tensor},
        outputs={"output": output_tensor}
    )