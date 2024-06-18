import os
from typing import List
import tensorflow as tf
from tfx import v1 as tfx
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import DataAccessor
from tfx.components.trainer.fn_args_utils import FnArgs
from tfx_bsl.tfxio import dataset_options

_TRAIN_DATA_SIZE = 50000
_EVAL_DATA_SIZE = 10000
_TRAIN_BATCH_SIZE = 64 # 80
_EVAL_BATCH_SIZE = 64 #80
_CLASSIFIER_LEARNING_RATE = 3e-4
_FINETUNE_LEARNING_RATE = 5e-5
_CLASSIFIER_EPOCHS = 12
_BUFFER_SIZE = 10000
_BATCH_SIZE = 64

# _TRAIN_DATA_SIZE = 128
# _EVAL_DATA_SIZE = 128
# _TRAIN_BATCH_SIZE = 32
# _EVAL_BATCH_SIZE = 32
# _CLASSIFIER_LEARNING_RATE = 1e-3
# _FINETUNE_LEARNING_RATE = 7e-6
# _CLASSIFIER_EPOCHS = 30

_IMAGE_KEY = 'image'
_LABEL_KEY = 'label'


def preprocessing_fn(inputs):
  """tf.transform's callback function for pre-processing inputs.

  Args:
    inputs: map from feature keys to raw not-yet-transformed features.

  Returns:
    Map from string feature key to transformed feature operations.
  """
  outputs = {}

  raw_image_dataset = inputs[_IMAGE_KEY]

  image_features = tf.map_fn(
      lambda x: tf.io.decode_png(x[0], channels=3),
      raw_image_dataset,
      dtype=tf.uint8)
  image_features = tf.image.resize(image_features, [32, 32])
  image_features = tf.cast(image_features, tf.float32)
  image_features /= 255.0

  outputs[_IMAGE_KEY] = image_features
  outputs[_LABEL_KEY] = inputs[_LABEL_KEY]

  return outputs


def _input_fn(file_pattern: List[str],
              data_accessor: DataAccessor,
              tf_transform_output: tft.TFTransformOutput,
              is_train: bool = False,
              batch_size: int = _BATCH_SIZE) -> tf.data.Dataset:
  """Generates features and label for tuning/training.

  Args:
    file_pattern: List of paths or patterns of input tfrecord files.
    data_accessor: DataAccessor for converting input to RecordBatch.
    tf_transform_output: A TFTransformOutput.
    is_train: Whether the input dataset is train split or not.
    batch_size: representing the number of consecutive elements of returned
      dataset to combine in a single batch

  Returns:
    A dataset that contains (features, indices) tuple where features is a
      dictionary of Tensors, and indices is a single Tensor of label indices.
  """
  dataset = data_accessor.tf_dataset_factory(
      file_pattern,
      dataset_options.TensorFlowDatasetOptions(
          batch_size=batch_size,
          label_key=_LABEL_KEY),
      tf_transform_output.transformed_metadata.schema)

  return dataset


def _build_keras_model():
    return tf.keras.Sequential([
        tf.keras.layers.InputLayer(
            input_shape=(32, 32, 3), name=_IMAGE_KEY),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(16, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])


# TFX Trainer will call this function.
def run_fn(fn_args: tfx.components.FnArgs):
  """Train the model based on given args.

  Args:
    fn_args: Holds args used to train the model as name/value pairs.

  Raises:
    ValueError: if invalid inputs.
  """
  tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

  train_dataset = _input_fn(
      fn_args.train_files,
      fn_args.data_accessor,
      tf_transform_output,
      is_train=True,
      batch_size=_TRAIN_BATCH_SIZE)

  eval_dataset = _input_fn(
      fn_args.eval_files,
      fn_args.data_accessor,
      tf_transform_output,
      is_train=False,
      batch_size=_EVAL_BATCH_SIZE)

  model = _build_keras_model()

  model.compile(
      loss=tf.keras.losses.sparse_categorical_crossentropy,
      optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
      metrics=['accuracy'])

  checkpoint_dir = './checkpoints'
  checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
  callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)]

  model.fit(
      train_dataset,
      steps_per_epoch=fn_args.train_steps,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps,
      callbacks=callbacks)

  model.save(fn_args.serving_model_dir, save_format='tf')

