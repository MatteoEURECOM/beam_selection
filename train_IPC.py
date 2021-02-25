import sys

import tensorflow as tf
from tensorflow.keras import losses, optimizers
import numpy as np
#from keras_flops import get_flops
import scipy.io as sio
import tensorflow.keras.backend as K

from dataloader import LidarDataset2D, LidarDataset3D
from models import Lidar2D

#def get_flops(model):
 #   run_meta = tf.RunMetadata()
  #  opts = tf.profiler.ProfileOptionBuilder.float_operation()
 
    # We use the Keras session graph in the call to the profiler.
   # flops = tf.profiler.profile(graph=K.get_session().graph, run_meta=run_meta, cmd='op', options=opts)
 
    #return flops.total_float_ops  # Prints the "flops" of the model.

if __name__ == '__main__':
    lidar_training_path = ["lidar_input_train.npz",
                           "lidar_input_validation.npz"]
    beam_training_path = ["beams_output_train.npz",
                          "beams_output_validation.npz"]
    
    training_data = LidarDataset2D(lidar_training_path, beam_training_path)

    lidar_test_path = ["lidar_input_test_s010.npz"]
    beam_test_path = ["beams_output_test_s010.npz"]
    
    test_data = LidarDataset2D(lidar_test_path, beam_test_path)

    training_data.lidar_data = np.transpose(training_data.lidar_data, (0, 2, 3, 1))
    test_data.lidar_data = np.transpose(test_data.lidar_data, (0, 2, 3, 1))
    
    y = training_data.beam_output
    yy = test_data.beam_output
    sio.savemat('labeltrain.mat',{'labeltrain':y})
    sio.savemat('labeltest.mat',{'labeltest':yy})

    model = Lidar2D
    loss_fn = lambda y_true, y_pred: -tf.reduce_sum(tf.reduce_mean(y_true[y_pred>0] * tf.math.log(y_pred[y_pred>0]), axis=0))

    top1 = tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='top_1_categorical_accuracy', dtype=None)
    top10 = tf.keras.metrics.TopKCategoricalAccuracy(k=10, name='top_10_categorical_accuracy', dtype=None)
    optim = optimizers.Adam(lr=1e-3, epsilon=1e-8)

    #scheduler = lambda epoch, lr: lr if epoch < 10 else lr/10
    def scheduler(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return lr/10

    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)



    model.compile(optimizer=optim, loss=loss_fn, metrics=[top1, top10])
    model.summary()
    #print(get_flops(model))
    model.fit(training_data.lidar_data, training_data.beam_output, callbacks=[callback], batch_size=16, epochs=20, validation_data=(test_data.lidar_data, test_data.beam_output))
    print(model.evaluate(test_data.lidar_data, test_data.beam_output, verbose=0))
    
    # Calculate throughput ratio
    #test_preds = model.predict(test_data.lidar_data, batch_size=100)
    #test_preds_idx = np.argsort(test_preds, axis=1)
    #print(np.sum(np.take_along_axis(test_data.beam_output_true, test_preds_idx, axis=1)[:, -1])/np.sum(np.max(test_data.beam_output_true, axis=1)))

    #top_k = np.zeros(100)
    #throughput_ratio_at_k = np.zeros(100)
    #correct = 0
    #for i in range(100):
     #   correct += np.sum(test_preds_idx[:, -1-i] == np.argmax(test_data.beam_output, axis=1))
      #  top_k[i] = correct/test_data.beam_output.shape[0]
       # throughput_ratio_at_k[i] = np.sum(np.log2(np.max(np.take_along_axis(test_data.beam_output_true, test_preds_idx, axis=1)[:, -1-i:], axis=1) + 1.0))/\
         #                          np.sum(np.log2(np.max(test_data.beam_output_true, axis=1) + 1.0))

    # print(top_k)
    # print(throughput_ratio_at_k)
    #sio.savemat('proposed_accuracy.mat',{'accuracy':top_k})
    #sio.savemat('proposed_throughput.mat',{'throughput':throughput_ratio_at_k})
    #np.savez("proposed.npz", classification=top_k, throughput_ratio=throughput_ratio_at_k)
