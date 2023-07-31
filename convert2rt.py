import tensorflow as tf
#from tensorflow.python.compiler.tensorrt import trt_convert as trt
import tensorflow.contrib.tensorrt as trt


# all_var = tf.global_variables()
# var_to_restore = []
# for var in all_var:
#     if 'Momentum' in var.name or 'LearnRate' in var.name:
#         continue
#     var_to_restore.append(var)

def calib_input_fn():
    input_size = (64, 96)
    return {"import/Placeholder_3": tf.placeholder(tf.float32, [None, input_size[1], input_size[1], 3])}


# saver = tf.train.Saver(var_to_restore)

# saver.restore(sess, FLAGS.model_path)

graph = tf.Graph()

output_nodes = ["truediv_3"]

with graph.as_default():
    with tf.Session() as sess:
        # First create a `Saver` object (for saving and rebuilding a         # model) and import your `MetaGraphDef` protocol buffer into it:         
        saver = tf.train.import_meta_graph("./data/models/model19.ckpt.meta")
        # Then restore your training data from checkpoint files:         
        saver.restore(sess, "./data/models/model19.ckpt")#.data-00000-of-00001")
        # Finally, freeze the graph:                your_outputs = [“your_output_node_names”]
        ################ TODO: print all internal node names 
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            sess,
            tf.get_default_graph().as_graph_def(),
            output_node_names=output_nodes)
        #graph_def = tf.GraphDef()
        #graph_def.ParseFromString(f.read())
        # Now you can create a TensorRT inference graph from your         # frozen graph:         
        trt_graph = trt.create_inference_graph(
            input_graph_def=frozen_graph,
            outputs=output_nodes,
            max_batch_size=1,
            max_workspace_size_bytes=100_000_000, # 1GB memory 
            precision_mode="FP16")
        # Import the TensorRT graph into a new graph and run:         
        output_node = tf.import_graph_def(
            trt_graph,
            return_elements=output_nodes)
        sess.run(output_node,
                 feed_dict=calib_input_fn())