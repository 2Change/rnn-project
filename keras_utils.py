def set_keras_session(gpu_memory_fraction=0.4):
    
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
    set_session(tf.Session(config=config))
