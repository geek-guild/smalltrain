from pathlib import Path
import os
import numpy as np

import tensorflow.compat.v1 as tf_v1 # TensorFlow 1.X
# import tensorflow as tf # TensorFlow 2.X
# tf.disable_eager_execution()

import onnx
import tf2onnx
from onnx_tf.backend import prepare

def save_to_onnx(sess, save_file_path,
                 input_names,
                 output_names):

    gd = sess.graph.as_graph_def()
    frozen_graph_def = tf_v1.graph_util.convert_variables_to_constants(sess, gd,
                                                                    [n.replace(':0', '') for n in output_names])

    # debug
    work_pb_file_path = os.path.join('/workspace', 'debug_saved_model.pbtxt')
    tf_v1.train.write_graph(frozen_graph_def, '.', work_pb_file_path, as_text=True)
    print('work_pb_file_path: {}'.format(work_pb_file_path))

    _graph = tf_v1.Graph()
    with _graph.as_default():
        tf_v1.import_graph_def(frozen_graph_def)
        input_names = ['import/{}'.format(s) for s in input_names]
        output_names = ['import/{}'.format(s) for s in output_names]
        onnx_graph = tf2onnx.tfonnx.process_tf_graph(_graph,
                                                     input_names=input_names,
                                                     output_names=output_names)
        model_proto = onnx_graph.make_model('smalltrain')
        print('save_to_onnx save_file_path: {}'.format(save_file_path))
        with open(save_file_path, 'wb') as f:
            f.write(model_proto.SerializeToString())

def load_from_onnx(model_path, input_data=None, expecred_values=None):

    print("load graph")

    model = onnx.load(model_path)

    tf_rep = prepare(model)

    print('----- inputs -----')
    print(tf_rep.inputs)
    print('-' * 10)
    print('----- outputs -----')
    print(tf_rep.outputs)
    print('-' * 10)

    _estimated = prepare(model).run(input_data)  # run the loaded model
    _estimated_label = np.argmax(_estimated, axis=1)
    print('_estimated_label: {}, expecred_values: {}'.format(_estimated_label, expecred_values))

    print("DONE load_model")
