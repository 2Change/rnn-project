import numpy as np
from im_utils import imresize


def tfnet_load(network='yolo', use_gpu=True):

    from darkflow.net.build import TFNet
    
    options = {"model": "cfg/{}.cfg".format(network), "load": "bin/{}.weights".format(network), "threshold": 0.1}
    print(options)
    if use_gpu:
        options["gpu"] = 0.8
        
    return TFNet(options)


def tfnet_predict(tfnet, im):
    
    if len(im.shape) == 3:          # just one image
        im = np.expand_dims(im, 0)
    
    this_inp = imresize(im, (608, 608), mode='RGB')  # 608x608 is Yolo input size
    this_inp = this_inp / 255.
    
    feed_dict = {tfnet.inp : this_inp}
    
    return tfnet.sess.run(tfnet.out, feed_dict)


def get_boxes(tfnet, yolo_out, h, w):
    
    for out in yolo_out:
    
        boxes = tfnet.framework.findboxes(out)
        threshold = tfnet.FLAGS.threshold
        boxesInfo = list()
        for box in boxes:
            tmpBox = tfnet.framework.process_box(box, h, w, threshold)
            if tmpBox is None:
                continue
            boxesInfo.append({
                "label": tmpBox[4],
                "confidence": tmpBox[6],
                "topleft": {
                    "x": tmpBox[0],
                    "y": tmpBox[2]},
                "bottomright": {
                    "x": tmpBox[1],
                    "y": tmpBox[3]}
            })
    
        yield boxesInfo
    
    


if __name__ == '__main__':
    
    import cv2
    import os
    
    rav = cv2.imread('../darkflow/ravaglia.jpg')
    rav = cv2.cvtColor(rav, cv2.COLOR_BGR2RGB)
    h, w, _ = rav.shape

    
    tfnet = tfnet_load('yolo')
    
    yolo_out = tfnet_predict(tfnet, rav)
    
    print(yolo_out.shape)
    
    images = []
    images_path = 'datasets/UCF11/separate_frames_50_h_240_w_320/train/basketball/v_shooting_01_01'
    for image_filename in sorted(os.listdir(images_path)):
        image = cv2.imread(os.path.join(images_path, image_filename))
        images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    images = np.array(images)
    yolo_ucf = tfnet_predict(tfnet, images)
    for box in get_boxes(yolo_ucf, 240, 320):
        print(box)
        print()
    
    #print(get_boxes(yolo_out, h, w))
    
    #print('Using return_predict method...')
    #print(tfnet.return_predict(cv2.imread('../darkflow/ravaglia.jpg')))
    
    file_path = 'datasets/UCF11/separate_frames_50_h_240_w_320_yolo_padding_False/train/basketball/v_shooting_01_01.npy'
    yolo_ucf = np.load(file_path)
    
    for box in get_boxes(yolo_ucf, 240, 320):
        print(box)
        print()