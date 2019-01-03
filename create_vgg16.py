import caffe
from caffe import layers as L, params as P
from caffe.proto import caffe_pb2

def create_vgg16(input_shape, classes=1000, deploy=False):
    net_name = "vgg16"
    data_root_dir = "/home/tim/datasets/cifar10/"
    if deploy:
        net_filename = "{0}_deploy.prototxt".format(net_name)
    else:
        net_filename = "{0}_train_test.prototxt".format(net_name)

    # net name
    with open(net_filename, "w") as f:
        f.write('name: "{0}"\n'.format(net_name))

    if deploy:
        net = caffe.NetSpec()
        """
        The conventional blob dimensions for batches of image data are 
        number N x channel K x height H x width W. Blob memory is row-major in layout, 
        so the last / rightmost dimension changes fastest. 
        For example, in a 4D blob, the value at index (n, k, h, w) is 
        physically located at index ((n * K + k) * H + h) * W + w.
        """
        # batch_size, channel, height, width
        net.data = L.Input(input_param=dict(shape=[dict(dim=list(input_shape))]))
    else:
        net = caffe.NetSpec()
        batch_size = 256
        lmdb = data_root_dir + "train_lmdb"
        net.data, net.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
                                     transform_param=dict(mirror=True,
                                                          crop_size=32,
                                                          mean_file=data_root_dir + "mean.binaryproto"),
                                     ntop=2, include=dict(phase=caffe_pb2.Phase.Value("TRAIN")))

        with open(net_filename, "a") as f:
            f.write(str(net.to_proto()))

        del net
        net = caffe.NetSpec()
        batch_size = 100
        lmdb = data_root_dir + "test_lmdb"
        net.data, net.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
                                     transform_param=dict(mirror=False,
                                                          crop_size=32,
                                                          mean_file=data_root_dir + "mean.binaryproto"),
                                     ntop=2, include=dict(phase=caffe_pb2.Phase.Value("TEST")))

    # Block 1
    # padding = 'same', equal to pad = 1
    net.block1_conv1 = L.Convolution(net.data, kernel_size=3, num_output=64, pad=1,
                              weight_filler=dict(type="xavier"),
                              bias_filler=dict(type="constant", value=0),
                              param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    net.block1_relu1 = L.ReLU(net.block1_conv1, in_place=True)
    net.block1_conv2 = L.Convolution(net.block1_relu1, kernel_size=3, num_output=64, pad=1,
                                     weight_filler=dict(type="xavier"),
                                     bias_filler=dict(type="constant", value=0),
                                     param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    net.block1_relu2 = L.ReLU(net.block1_conv2, in_place=True)
    net.block1_pool = L.Pooling(net.block1_relu2, kernel_size=2, stride=2, pool = P.Pooling.MAX)

    # Block 2
    net.block2_conv1 = L.Convolution(net.block1_pool, kernel_size=3, num_output=128, pad=1,
                                     weight_filler=dict(type="xavier"),
                                     bias_filler=dict(type="constant", value=0),
                                     param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    net.block2_relu1 = L.ReLU(net.block2_conv1, in_place=True)
    net.block2_conv2 = L.Convolution(net.block2_relu1, kernel_size=3, num_output=128, pad=1,
                                     weight_filler=dict(type="xavier"),
                                     bias_filler=dict(type="constant", value=0),
                                     param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    net.block2_relu2 = L.ReLU(net.block2_conv2, in_place=True)
    net.block2_pool = L.Pooling(net.block2_relu2, kernel_size=2, stride=2, pool=P.Pooling.MAX)

    # Block 3
    net.block3_conv1 = L.Convolution(net.block2_pool, kernel_size=3, num_output=256, pad=1,
                                     weight_filler=dict(type="xavier"),
                                     bias_filler=dict(type="constant", value=0),
                                     param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    net.block3_relu1 = L.ReLU(net.block3_conv1, in_place=True)
    net.block3_conv2 = L.Convolution(net.block3_relu1, kernel_size=3, num_output=256, pad=1,
                                     weight_filler=dict(type="xavier"),
                                     bias_filler=dict(type="constant", value=0),
                                     param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    net.block3_relu2 = L.ReLU(net.block3_conv2, in_place=True)
    net.block3_conv3 = L.Convolution(net.block3_relu2, kernel_size=3, num_output=256, pad=1,
                                     weight_filler=dict(type="xavier"),
                                     bias_filler=dict(type="constant", value=0),
                                     param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    net.block3_relu3 = L.ReLU(net.block3_conv3, in_place=True)
    net.block3_pool = L.Pooling(net.block3_relu3, kernel_size=2, stride=2, pool=P.Pooling.MAX)

    # Block 4
    net.block4_conv1 = L.Convolution(net.block3_pool, kernel_size=3, num_output=512, pad=1,
                                     weight_filler=dict(type="xavier"),
                                     bias_filler=dict(type="constant", value=0),
                                     param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    net.block4_relu1 = L.ReLU(net.block4_conv1, in_place=True)
    net.block4_conv2 = L.Convolution(net.block4_relu1, kernel_size=3, num_output=512, pad=1,
                                     weight_filler=dict(type="xavier"),
                                     bias_filler=dict(type="constant", value=0),
                                     param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    net.block4_relu2 = L.ReLU(net.block4_conv2, in_place=True)
    net.block4_conv3 = L.Convolution(net.block4_relu2, kernel_size=3, num_output=512, pad=1,
                                     weight_filler=dict(type="xavier"),
                                     bias_filler=dict(type="constant", value=0),
                                     param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    net.block4_relu3 = L.ReLU(net.block4_conv3, in_place=True)
    net.block4_pool = L.Pooling(net.block4_relu3, kernel_size=2, stride=2, pool=P.Pooling.MAX)

    # Block 5
    net.block5_conv1 = L.Convolution(net.block4_pool, kernel_size=3, num_output=512, pad=1,
                                     weight_filler=dict(type="xavier"),
                                     bias_filler=dict(type="constant", value=0),
                                     param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    net.block5_relu1 = L.ReLU(net.block5_conv1, in_place=True)
    net.block5_conv2 = L.Convolution(net.block5_relu1, kernel_size=3, num_output=512, pad=1,
                                     weight_filler=dict(type="xavier"),
                                     bias_filler=dict(type="constant", value=0),
                                     param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    net.block5_relu2 = L.ReLU(net.block5_conv2, in_place=True)
    net.block5_conv3 = L.Convolution(net.block5_relu2, kernel_size=3, num_output=512, pad=1,
                                     weight_filler=dict(type="xavier"),
                                     bias_filler=dict(type="constant", value=0),
                                     param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    net.block5_relu3 = L.ReLU(net.block5_conv3, in_place=True)
    net.block5_pool = L.Pooling(net.block5_relu3, kernel_size=2, stride=2, pool=P.Pooling.MAX)

    # Classification block
    net.cf_block_fc1 = L.InnerProduct(net.block5_pool, num_output=4096,
                             weight_filler=dict(type="xavier"),
                             bias_filler=dict(type="constant", value=0),
                             param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    net.cf_block_relu1 = L.ReLU(net.cf_block_fc1, in_place=True)

    net.cf_block_fc2 = L.InnerProduct(net.cf_block_relu1, num_output=4096,
                                      weight_filler=dict(type="xavier"),
                                      bias_filler=dict(type="constant", value=0),
                                      param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    net.cf_block_relu2 = L.ReLU(net.cf_block_fc2, in_place=True)

    net.cf_block_pred_fc = L.InnerProduct(net.cf_block_relu2, num_output=classes,
                                      weight_filler=dict(type="xavier"),
                                      bias_filler=dict(type="constant", value=0),
                                      param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])

    if deploy:
        net.prob = L.Softmax(net.cf_block_pred_fc)
    else:
        net.accuracy = L.Accuracy(net.cf_block_pred_fc, net.label,
                                  include=dict(phase=caffe_pb2.Phase.Value('TEST')))

        net.loss = L.SoftmaxWithLoss(net.cf_block_pred_fc, net.label)

    with open(net_filename, "a") as f:
        f.write(str(net.to_proto()))

if __name__ == "__main__":

    input_shape = [1, 3, 32, 32]
    classes = 10

    create_vgg16(input_shape=input_shape, classes=classes, deploy=False)
    create_vgg16(input_shape=input_shape, classes=classes, deploy=True)

    net_name = "vgg16"

    solver = caffe.SGDSolver("{0}_solver.prototxt".format(net_name))

    for k, v in solver.net.blobs.items():
        print(k, v.data.shape)