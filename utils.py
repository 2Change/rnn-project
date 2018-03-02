from PIL import Image
import numpy as np

def preprocess_images_tf(images):
    
    """
        Change the range of an image np array,
        from (0,255) to (-1,1).
        Needs to be used for InceptionV3 network.
    """
    
    return ((images / 255.) - 0.5) * 2


def take(generator, how_many):
    
    """
    Take the first how_many results from a generator 
    (or less, if the generator won't generate as many results).
    Note that this function is also a generator.
    """
    
    for _, res in zip(range(how_many), generator):
        yield res
        
        
        
def load_image_as_np_array(image_path):
    
    # Pillow returns images as integers in range (0, 255)
    
    image = Image.open(image_path)
    return np.array(image)