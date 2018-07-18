import os
import sys
from pprint import pprint

import winsound
import numpy as np
import pandas as pd
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)
directory = 'kinect_head_pose_db/hpdb/'
cropped_directory = 'kinect_head_pose_db/cropped_rgb/'


def get_predict_data():
    image_string = tf.read_file(sys.argv[1])
    image_decoded = tf.image.decode_png(image_string)
    image_resized = tf.image.resize_images(image_decoded, [64, 64])
    dataset = tf.data.Dataset.from_tensors(image_resized)
    return dataset


def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_png(image_string)
    image_resized = tf.image.resize_images(image_decoded, [64, 64])  # tf.cast(image_decoded, dtype=tf.float64)
    return image_resized, label


def get_datasets_train():
    print("Getting Train-Data!")
    filenames = []
    labels = []
    folder_counter = sum([len(d) for r, d, folder in os.walk(directory)])
    for i in range(1, folder_counter - 22):
        print("i" + str(i))
        subdirect = directory + '{:02}'.format(i) + "/"
        cropped_subdirect = cropped_directory + '{:02}'.format(i) + "/"

        try:
            for filename in os.listdir(subdirect):
                if filename.endswith("_pose.txt"):
                    labels.append(np.reshape(
                        pd.read_csv(os.path.join(subdirect, filename), delimiter=" ", header=None, nrows=3).dropna(
                            axis=1).values, [-1]))
                    continue
                if filename.endswith(".png"):
                    filenames.append(os.path.join(cropped_subdirect, filename))
                    continue
                if filename.endswith("_depth.bin"):
                    pass
                else:
                    # print("Lets Continue")
                    continue
        except Exception:
            print("Folder not found")
            continue
    batch_size = 100
    filename_tensor = tf.constant(filenames)
    labels_tensor = tf.constant(np.array(labels))

    dataset = tf.data.Dataset.from_tensor_slices((filename_tensor, labels_tensor))

    dataset = dataset.apply(tf.contrib.data.map_and_batch(map_func=_parse_function, batch_size=batch_size))

    dataset = dataset.repeat()
    iterator = dataset.make_one_shot_iterator()

    features, labels = iterator.get_next()
    return features, labels


def get_datasets_test():
    print("Getting Test-Data!")
    filenames = []
    labels = []
    folder_counter = sum([len(d) for r, d, folder in os.walk(directory)])
    for i in range(folder_counter - 4, folder_counter):
        print("i" + str(i))
        subdirect = directory + '{:02}'.format(i) + "/"
        try:
            for filename in os.listdir(subdirect):
                if filename.endswith("_pose.txt"):
                    labels.append(np.reshape(
                        pd.read_csv(os.path.join(subdirect, filename), delimiter=" ", header=None, nrows=3).dropna(
                            axis=1).values, [-1]))
                    continue
                if filename.endswith(".png"):
                    filenames.append(os.path.join(subdirect, filename))
                    continue
                if filename.endswith("_depth.bin"):
                    pass
                else:
                    # print("Lets Continue")
                    continue
        except Exception:
            print("Folder not found")
            continue
    batch_size = 100
    filename_tensor = tf.constant(filenames)
    labels_tensor = tf.constant(np.array(labels))

    dataset = tf.data.Dataset.from_tensor_slices((filename_tensor, labels_tensor))
    dataset = dataset.map(_parse_function)
    dataset = dataset.batch(batch_size)
    return dataset


def get_datasets_eval():
    print("Getting Eval-Data!")
    filenames = []
    labels = []
    folder_counter = sum([len(d) for r, d, folder in os.walk(directory)])
    for i in range(folder_counter, folder_counter + 1):
        print("i" + str(i))
        subdirect = directory + '{:02}'.format(i) + "/"
        try:
            for filename in os.listdir(subdirect):
                if filename.endswith("_pose.txt"):
                    labels.append(np.reshape(
                        pd.read_csv(os.path.join(subdirect, filename), delimiter=" ", header=None, nrows=3).dropna(
                            axis=1).values, [-1]))
                    continue
                if filename.endswith(".png"):
                    filenames.append(os.path.join(subdirect, filename))
                    continue
                if filename.endswith("_depth.bin"):
                    pass
                else:
                    # print("Lets Continue")
                    continue
        except Exception:
            print("Folder not found")
            continue
    batch_size = 100
    filename_tensor = tf.constant(filenames)
    labels_tensor = tf.constant(np.array(labels))

    dataset = tf.data.Dataset.from_tensor_slices((filename_tensor, labels_tensor))
    dataset = dataset.map(_parse_function)
    dataset = dataset.batch(batch_size)
    return dataset


def cnn_model_fn(features, labels, mode):
    input_layer = tf.reshape(features, [-1, 64, 64, 3])

    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=30,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.tanh
    )
    maxp1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    conv2 = tf.layers.conv2d(
        inputs=maxp1,
        filters=30,
        kernel_size=[4, 4],
        padding="same",
        activation=tf.tanh
    )
    maxp2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    conv3 = tf.layers.conv2d(
        inputs=maxp2,
        filters=30,
        kernel_size=[4, 4],
        padding="same",
        activation=tf.tanh
    )

    maxp3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

    conv4 = tf.layers.conv2d(
        inputs=maxp3,
        filters=30,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.tanh
    )

    conv5 = tf.layers.conv2d(
        inputs=conv4,
        filters=120,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.tanh
    )

    unit_count = 8 * 8 * 120
    flattened = tf.reshape(conv5, [-1, unit_count])
    dense = tf.layers.dense(inputs=flattened, units=unit_count, activation=tf.tanh)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    full_layer2 = tf.layers.dense(inputs=dropout, units=84, activation=tf.tanh)
    logits = tf.layers.dense(inputs=full_layer2, units=9)

    predictions = {'Predictions': tf.reshape(tensor=logits, shape=[-1, 3, 3], name="shaped_tensor")}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.mean_squared_error(labels=labels, predictions=logits)

    starter_learning_rate = 0.001
    learning_rate = tf.train.exponential_decay(learning_rate=starter_learning_rate,
                                               global_step=tf.train.get_global_step(),
                                               decay_steps=10000,
                                               decay_rate=0.1,
                                               staircase=True)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        "Mean Absolute Error": tf.metrics.mean_absolute_error(labels=labels,
                                                              predictions=tf.cast(logits, dtype=tf.float64))}

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused):
    print('Create the Estimator')
    head_pose_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        model_dir="tmp/model")
    if len(sys.argv) <= 1:
        # print('Set up logging for predictions')
        # print('Log the values in the "Sigmoid" tensor with label "probabilities"')
        tensors_to_log = {"Predictions": "shaped_tensor"}
        logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)
        print('Train the model')
        head_pose_classifier.train(
            input_fn=get_datasets_train,
            max_steps=40000,
            hooks=[logging_hook]
        )

        print("Evaluate the Model")
        eval_results = head_pose_classifier.evaluate(input_fn=get_datasets_test)
        print(eval_results)
        winsound.Beep(1500, 25)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".png"):
        print("Predict the Rotation!")
        prediction = head_pose_classifier.predict(input_fn=get_predict_data)
        for p in prediction:
            pprint(p)
    if len(sys.argv) > 2:
        print("ERROR:To Many Argumts were given or not a PNG!")


if __name__ == "__main__":
    tf.app.run(main)
