import tensorflow as tf

tf_major_version = int(tf.__version__.split('.')[0])
print('tf.__version__: {}'.format(tf.__version__))
print('tf_major_version: {}'.format(tf_major_version))
if tf_major_version >= 2:
    print('from tensorflow_addons.image import rotate')
    from tensorflow_addons.image import rotate
elif tf_major_version >= 1 and tf_major_version < 2:
    print('import tensorflow.contrib.image.rotate as rotate')
    from tensorflow.contrib.image import rotate
else:
    raise ImportError('TensorFlow with major version {} is not available for SmallTrain')


