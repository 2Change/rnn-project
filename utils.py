
def preprocess_images_tf(images):
    
    """
        Change the range of an image np array,
        from (0,255) to (-1,1).
        Needs to be used for InceptionV3 network.
    """
    
    return ((images / 255.) - 0.5) * 2
