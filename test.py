from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from utils import tfrecord_voc_utils as voc_utils
import tensorflow as tf
import numpy as np
import RetinaNet as net
import os
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from skimage import io, transform
from utils.voc_classname_encoder import classname_to_ids
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

lr = 0.01
batch_size = 1
buffer_size = 20
epochs = 280
input_shape = [500, 500 ,3]
reduce_lr_epoch = [120, 250]
config = {
    'is_bottleneck': True,
    'residual_block_list': [3, 4, 6, 3],
    'init_conv_filters': 16,
    'init_conv_kernel_size': 3,
    'init_conv_strides': 1,
    'init_pooling_pool_size': 3,
    'init_pooling_strides': 2,

    'mode': 'train',     # 'train', 'test'
    'is_pretraining': False,
    'data_shape': input_shape,
    'num_classes': 20,
    'weight_decay': 1e-4,
    'keep_prob': 0.5,
    'data_format': 'channels_last',
    'batch_size': batch_size,

    'nms_score_threshold': 0.2,
    'nms_max_boxes': 10,
    'nms_iou_threshold': 0.5,
}

image_preprocess_config = {
    'data_format': 'channels_last',
    'target_size': [500, 500],
    'shorter_side': 480,
    'is_random_crop': False,
    'random_horizontal_flip': 0.5,
    'random_vertical_flip': 0.,
    'pad_truth_to': 60
}

data = ['./test/test_00000-of-00005.tfrecord',
        './test/test_00001-of-00005.tfrecord']

train_gen = voc_utils.get_generator(data,
                                    batch_size, buffer_size, image_preprocess_config)
trainset_provider = {
    'data_shape': [500, 500, 3],
    'num_train': 100,
    'num_val': 0,
    'train_generator': train_gen,
    'val_generator': None
}
retinanet = net.RetinaNet(config, trainset_provider)
# retinanet.load_weight('./retinanet/test-64954')
for i in range(epochs):
    print('-'*25, 'epoch', i, '-'*25)
    if i in reduce_lr_epoch:
        lr = lr/10.
        print('reduce lr, lr=', lr, 'now')
    mean_loss = retinanet.train_one_epoch(lr)
    print('>> mean loss', mean_loss)
    retinanet.save_weight('latest', './retina/test')


# img = io.imread('000026.jpg')
# img = transform.resize(img, [300,300])
# img = np.expand_dims(img, 0)
# result = ssd300.test_one_image(img)
# id_to_clasname = {k:v for (v,k) in classname_to_ids.items()}
# scores = result[0]
# bbox = result[1]
# class_id = result[2]
# print(scores, bbox, class_id)
# plt.figure(1)
# plt.imshow(np.squeeze(img))
# axis = plt.gca()
# for i in range(len(scores)):
#     rect = patches.Rectangle((bbox[i][1],bbox[i][0]), bbox[i][3]-bbox[i][1],bbox[i][2]-bbox[i][0],linewidth=2,edgecolor='b',facecolor='none')
#     axis.add_patch(rect)
#     plt.text(bbox[i][1],bbox[i][0], id_to_clasname[class_id[i]]+str(' ')+str(scores[i]), color='red', fontsize=12)
# plt.show()
