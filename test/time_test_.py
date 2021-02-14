node {
  name: "data"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "data_bn/FusedBatchNorm"
  op: "FusedBatchNorm"
  input: "data:0"
  input: "data_bn/gamma"
  input: "data_bn/beta"
  input: "data_bn/mean"
  input: "data_bn/std"
  attr {
    key: "epsilon"
    value {
      f: 1.00099996416e-05
    }
  }
}
node {
  name: "data_scale/Mul"
  op: "Mul"
  input: "data_bn/FusedBatchNorm"
  input: "data_scale/mul"
}
node {
  name: "data_scale/BiasAdd"
  op: "BiasAdd"
  input: "data_scale/Mul"
  input: "data_scale/add"
}
node {
  name: "SpaceToBatchND/block_shape"
  op: "Const"
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        int_val: 1
        int_val: 1
      }
    }
  }
}
node {
  name: "SpaceToBatchND/paddings"
  op: "Const"
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
          dim {
            size: 2
          }
        }
        int_val: 3
        int_val: 3
        int_val: 3
        int_val: 3
      }
    }
  }
}
node {
  name: "Pad"
  op: "SpaceToBatchND"
  input: "data_scale/BiasAdd"
  input: "SpaceToBatchND/block_shape"
  input: "SpaceToBatchND/paddings"
}
node {
  name: "conv1_h/Conv2D"
  op: "Conv2D"
  input: "Pad"
  input: "conv1_h/weights"
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "VALID"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 2
        i: 2
        i: 1
      }
    }
  }
}
node {
  name: "conv1_h/BiasAdd"
  op: "BiasAdd"
  input: "conv1_h/Conv2D"
  input: "conv1_h/bias"
}
node {
  name: "BatchToSpaceND"
  op: "BatchToSpaceND"
  input: "conv1_h/BiasAdd"
}
node {
  name: "conv1_bn_h/FusedBatchNorm"
  op: "FusedBatchNorm"
  input: "BatchToSpaceND"
  input: "conv1_bn_h/gamma"
  input: "conv1_bn_h/beta"
  input: "conv1_bn_h/mean"
  input: "conv1_bn_h/std"
  attr {
    key: "epsilon"
    value {
      f: 1.00099996416e-05
    }
  }
}
node {
  name: "conv1_scale_h/Mul"
  op: "Mul"
  input: "conv1_bn_h/FusedBatchNorm"
  input: "conv1_scale_h/mul"
}
node {
  name: "conv1_scale_h/BiasAdd"
  op: "BiasAdd"
  input: "conv1_scale_h/Mul"
  input: "conv1_scale_h/add"
}
node {
  name: "Relu"
  op: "Relu"
  input: "conv1_scale_h/BiasAdd"
}
node {
  name: "conv1_pool/MaxPool"
  op: "MaxPool"
  input: "Relu"
  attr {
    key: "ksize"
    value {
      list {
        i: 1
        i: 3
        i: 3
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 2
        i: 2
        i: 1
      }
    }
  }
}
node {
  name: "layer_64_1_conv1_h/Conv2D"
  op: "Conv2D"
  input: "conv1_pool/MaxPool"
  input: "layer_64_1_conv1_h/weights"
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
}
node {
  name: "layer_64_1_bn2_h/FusedBatchNorm"
  op: "BiasAdd"
  input: "layer_64_1_conv1_h/Conv2D"
  input: "layer_64_1_conv1_h/Conv2D_bn_offset"
}
node {
  name: "layer_64_1_scale2_h/Mul"
  op: "Mul"
  input: "layer_64_1_bn2_h/FusedBatchNorm"
  input: "layer_64_1_scale2_h/mul"
}
node {
  name: "layer_64_1_scale2_h/BiasAdd"
  op: "BiasAdd"
  input: "layer_64_1_scale2_h/Mul"
  input: "layer_64_1_scale2_h/add"
}
node {
  name: "Relu_1"
  op: "Relu"
  input: "layer_64_1_scale2_h/BiasAdd"
}
node {
  name: "layer_64_1_conv2_h/Conv2D"
  op: "Conv2D"
  input: "Relu_1"
  input: "layer_64_1_conv2_h/weights"
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
}
node {
  name: "add"
  op: "Add"
  input: "layer_64_1_conv2_h/Conv2D"
  input: "conv1_pool/MaxPool"
}
node {
  name: "layer_128_1_bn1_h/FusedBatchNorm"
  op: "FusedBatchNorm"
  input: "add"
  input: "layer_128_1_bn1_h/gamma"
  input: "layer_128_1_bn1_h/beta"
  input: "layer_128_1_bn1_h/mean"
  input: "layer_128_1_bn1_h/std"
  attr {
    key: "epsilon"
    value {
      f: 1.00099996416e-05
    }
  }
}
node {
  name: "layer_128_1_scale1_h/Mul"
  op: "Mul"
  input: "layer_128_1_bn1_h/FusedBatchNorm"
  input: "layer_128_1_scale1_h/mul"
}
node {
  name: "layer_128_1_scale1_h/BiasAdd"
  op: "BiasAdd"
  input: "layer_128_1_scale1_h/Mul"
  input: "layer_128_1_scale1_h/add"
}
node {
  name: "Relu_2"
  op: "Relu"
  input: "layer_128_1_scale1_h/BiasAdd"
}
node {
  name: "layer_128_1_conv_expand_h/Conv2D"
  op: "Conv2D"
  input: "Relu_2"
  input: "layer_128_1_conv_expand_h/weights"
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 2
        i: 2
        i: 1
      }
    }
  }
}
node {
  name: "layer_128_1_conv1_h/Conv2D"
  op: "Conv2D"
  input: "Relu_2"
  input: "layer_128_1_conv1_h/weights"
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 2
        i: 2
        i: 1
      }
    }
  }
}
node {
  name: "layer_128_1_bn2/FusedBatchNorm"
  op: "BiasAdd"
  input: "layer_128_1_conv1_h/Conv2D"
  input: "layer_128_1_conv1_h/Conv2D_bn_offset"
}
node {
  name: "layer_128_1_scale2/Mul"
  op: "Mul"
  input: "layer_128_1_bn2/FusedBatchNorm"
  input: "layer_128_1_scale2/mul"
}
node {
  name: "layer_128_1_scale2/BiasAdd"
  op: "BiasAdd"
  input: "layer_128_1_scale2/Mul"
  input: "layer_128_1_scale2/add"
}
node {
  name: "Relu_3"
  op: "Relu"
  input: "layer_128_1_scale2/BiasAdd"
}
node {
  name: "layer_128_1_conv2/Conv2D"
  op: "Conv2D"
  input: "Relu_3"
  input: "layer_128_1_conv2/weights"
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
}
node {
  name: "add_1"
  op: "Add"
  input: "layer_128_1_conv2/Conv2D"
  input: "layer_128_1_conv_expand_h/Conv2D"
}
node {
  name: "layer_256_1_bn1/FusedBatchNorm"
  op: "FusedBatchNorm"
  input: "add_1"
  input: "layer_256_1_bn1/gamma"
  input: "layer_256_1_bn1/beta"
  input: "layer_256_1_bn1/mean"
  input: "layer_256_1_bn1/std"
  attr {
    key: "epsilon"
    value {
      f: 1.00099996416e-05
    }
  }
}
node {
  name: "layer_256_1_scale1/Mul"
  op: "Mul"
  input: "layer_256_1_bn1/FusedBatchNorm"
  input: "layer_256_1_scale1/mul"
}
node {
  name: "layer_256_1_scale1/BiasAdd"
  op: "BiasAdd"
  input: "layer_256_1_scale1/Mul"
  input: "layer_256_1_scale1/add"
}
node {
  name: "Relu_4"
  op: "Relu"
  input: "layer_256_1_scale1/BiasAdd"
}
node {
  name: "SpaceToBatchND_1/paddings"
  op: "Const"
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
          dim {
            size: 2
          }
        }
        int_val: 1
        int_val: 1
        int_val: 1
        int_val: 1
      }
    }
  }
}
node {
  name: "layer_256_1_conv_expand/Conv2D"
  op: "Conv2D"
  input: "Relu_4"
  input: "layer_256_1_conv_expand/weights"
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 2
        i: 2
        i: 1
      }
    }
  }
}
node {
  name: "conv4_3_norm/l2_normalize"
  op: "L2Normalize"
  input: "Relu_4:0"
  input: "conv4_3_norm/l2_normalize/Sum/reduction_indices"
}
node {
  name: "conv4_3_norm/mul_1"
  op: "Mul"
  input: "conv4_3_norm/l2_normalize"
  input: "conv4_3_norm/mul"
}
node {
  name: "conv4_3_norm_mbox_loc/Conv2D"
  op: "Conv2D"
  input: "conv4_3_norm/mul_1"
  input: "conv4_3_norm_mbox_loc/weights"
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
}
node {
  name: "conv4_3_norm_mbox_loc/BiasAdd"
  op: "BiasAdd"
  input: "conv4_3_norm_mbox_loc/Conv2D"
  input: "conv4_3_norm_mbox_loc/bias"
}
node {
  name: "flatten/Reshape"
  op: "Flatten"
  input: "conv4_3_norm_mbox_loc/BiasAdd"
}
node {
  name: "conv4_3_norm_mbox_conf/Conv2D"
  op: "Conv2D"
  input: "conv4_3_norm/mul_1"
  input: "conv4_3_norm_mbox_conf/weights"
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
}
node {
  name: "conv4_3_norm_mbox_conf/BiasAdd"
  op: "BiasAdd"
  input: "conv4_3_norm_mbox_conf/Conv2D"
  input: "conv4_3_norm_mbox_conf/bias"
}
node {
  name: "flatten_6/Reshape"
  op: "Flatten"
  input: "conv4_3_norm_mbox_conf/BiasAdd"
}
node {
  name: "Pad_1"
  op: "SpaceToBatchND"
  input: "Relu_4"
  input: "SpaceToBatchND/block_shape"
  input: "SpaceToBatchND_1/paddings"
}
node {
  name: "layer_256_1_conv1/Conv2D"
  op: "Conv2D"
  input: "Pad_1"
  input: "layer_256_1_conv1/weights"
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "VALID"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 2
        i: 2
        i: 1
      }
    }
  }
}
node {
  name: "layer_256_1_bn2/FusedBatchNorm"
  op: "BiasAdd"
  input: "layer_256_1_conv1/Conv2D"
  input: "layer_256_1_conv1/Conv2D_bn_offset"
}
node {
  name: "BatchToSpaceND_1"
  op: "BatchToSpaceND"
  input: "layer_256_1_bn2/FusedBatchNorm"
}
node {
  name: "layer_256_1_scale2/Mul"
  op: "Mul"
  input: "BatchToSpaceND_1"
  input: "layer_256_1_scale2/mul"
}
node {
  name: "layer_256_1_scale2/BiasAdd"
  op: "BiasAdd"
  input: "layer_256_1_scale2/Mul"
  input: "layer_256_1_scale2/add"
}
node {
  name: "Relu_5"
  op: "Relu"
  input: "layer_256_1_scale2/BiasAdd"
}
node {
  name: "layer_256_1_conv2/Conv2D"
  op: "Conv2D"
  input: "Relu_5"
  input: "layer_256_1_conv2/weights"
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
}
node {
  name: "add_2"
  op: "Add"
  input: "layer_256_1_conv2/Conv2D"
  input: "layer_256_1_conv_expand/Conv2D"
}
node {
  name: "layer_512_1_bn1/FusedBatchNorm"
  op: "FusedBatchNorm"
  input: "add_2"
  input: "layer_512_1_bn1/gamma"
  input: "layer_512_1_bn1/beta"
  input: "layer_512_1_bn1/mean"
  input: "layer_512_1_bn1/std"
  attr {
    key: "epsilon"
    value {
      f: 1.00099996416e-05
    }
  }
}
node {
  name: "layer_512_1_scale1/Mul"
  op: "Mul"
  input: "layer_512_1_bn1/FusedBatchNorm"
  input: "layer_512_1_scale1/mul"
}
node {
  name: "layer_512_1_scale1/BiasAdd"
  op: "BiasAdd"
  input: "layer_512_1_scale1/Mul"
  input: "layer_512_1_scale1/add"
}
node {
  name: "Relu_6"
  op: "Relu"
  input: "layer_512_1_scale1/BiasAdd"
}
node {
  name: "layer_512_1_conv_expand_h/Conv2D"
  op: "Conv2D"
  input: "Relu_6"
  input: "layer_512_1_conv_expand_h/weights"
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
}
node {
  name: "layer_512_1_conv1_h/Conv2D"
  op: "Conv2D"
  input: "Relu_6"
  input: "layer_512_1_conv1_h/weights"
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
}
node {
  name: "layer_512_1_bn2_h/FusedBatchNorm"
  op: "BiasAdd"
  input: "layer_512_1_conv1_h/Conv2D"
  input: "layer_512_1_conv1_h/Conv2D_bn_offset"
}
node {
  name: "layer_512_1_scale2_h/Mul"
  op: "Mul"
  input: "layer_512_1_bn2_h/FusedBatchNorm"
  input: "layer_512_1_scale2_h/mul"
}
node {
  name: "layer_512_1_scale2_h/BiasAdd"
  op: "BiasAdd"
  input: "layer_512_1_scale2_h/Mul"
  input: "layer_512_1_scale2_h/add"
}
node {
  name: "Relu_7"
  op: "Relu"
  input: "layer_512_1_scale2_h/BiasAdd"
}
node {
  name: "layer_512_1_conv2_h/convolution/SpaceToBatchND"
  op: "SpaceToBatchND"
  input: "Relu_7"
  input: "layer_512_1_conv2_h/convolution/SpaceToBatchND/block_shape"
  input: "layer_512_1_conv2_h/convolution/SpaceToBatchND/paddings"
}
node {
  name: "layer_512_1_conv2_h/convolution"
  op: "Conv2D"
  input: "layer_512_1_conv2_h/convolution/SpaceToBatchND"
  input: "layer_512_1_conv2_h/weights"
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "VALID"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
}
node {
  name: "layer_512_1_conv2_h/convolution/BatchToSpaceND"
  op: "BatchToSpaceND"
  input: "layer_512_1_conv2_h/convolution"
  input: "layer_512_1_conv2_h/convolution/BatchToSpaceND/block_shape"
  input: "layer_512_1_conv2_h/convolution/BatchToSpaceND/crops"
}
node {
  name: "add_3"
  op: "Add"
  input: "layer_512_1_conv2_h/convolution/BatchToSpaceND"
  input: "layer_512_1_conv_expand_h/Conv2D"
}
node {
  name: "last_bn_h/FusedBatchNorm"
  op: "FusedBatchNorm"
  input: "add_3"
  input: "last_bn_h/gamma"
  input: "last_bn_h/beta"
  input: "last_bn_h/mean"
  input: "last_bn_h/std"
  attr {
    key: "epsilon"
    value {
      f: 1.00099996416e-05
    }
  }
}
node {
  name: "last_scale_h/Mul"
  op: "Mul"
  input: "last_bn_h/FusedBatchNorm"
  input: "last_scale_h/mul"
}
node {
  name: "last_scale_h/BiasAdd"
  op: "BiasAdd"
  input: "last_scale_h/Mul"
  input: "last_scale_h/add"
}
node {
  name: "last_relu"
  op: "Relu"
  input: "last_scale_h/BiasAdd"
}
node {
  name: "conv6_1_h/Conv2D"
  op: "Conv2D"
  input: "last_relu"
  input: "conv6_1_h/weights"
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
}
node {
  name: "conv6_1_h/BiasAdd"
  op: "BiasAdd"
  input: "conv6_1_h/Conv2D"
  input: "conv6_1_h/bias"
}
node {
  name: "conv6_1_h/Relu"
  op: "Relu"
  input: "conv6_1_h/BiasAdd"
}
node {
  name: "conv6_2_h/Conv2D"
  op: "Conv2D"
  input: "conv6_1_h/Relu"
  input: "conv6_2_h/weights"
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 2
        i: 2
        i: 1
      }
    }
  }
}
node {
  name: "conv6_2_h/BiasAdd"
  op: "BiasAdd"
  input: "conv6_2_h/Conv2D"
  input: "conv6_2_h/bias"
}
node {
  name: "conv6_2_h/Relu"
  op: "Relu"
  input: "conv6_2_h/BiasAdd"
}
node {
  name: "conv7_1_h/Conv2D"
  op: "Conv2D"
  input: "conv6_2_h/Relu"
  input: "conv7_1_h/weights"
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
}
node {
  name: "conv7_1_h/BiasAdd"
  op: "BiasAdd"
  input: "conv7_1_h/Conv2D"
  input: "conv7_1_h/bias"
}
node {
  name: "conv7_1_h/Relu"
  op: "Relu"
  input: "conv7_1_h/BiasAdd"
}
node {
  name: "Pad_2"
  op: "SpaceToBatchND"
  input: "conv7_1_h/Relu"
  input: "SpaceToBatchND/block_shape"
  input: "SpaceToBatchND_1/paddings"
}
node {
  name: "conv7_2_h/Conv2D"
  op: "Conv2D"
  input: "Pad_2"
  input: "conv7_2_h/weights"
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "VALID"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 2
        i: 2
        i: 1
      }
    }
  }
}
node {
  name: "conv7_2_h/BiasAdd"
  op: "BiasAdd"
  input: "conv7_2_h/Conv2D"
  input: "conv7_2_h/bias"
}
node {
  name: "BatchToSpaceND_2"
  op: "BatchToSpaceND"
  input: "conv7_2_h/BiasAdd"
}
node {
  name: "conv7_2_h/Relu"
  op: "Relu"
  input: "BatchToSpaceND_2"
}
node {
  name: "conv8_1_h/Conv2D"
  op: "Conv2D"
  input: "conv7_2_h/Relu"
  input: "conv8_1_h/weights"
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
}
node {
  name: "conv8_1_h/BiasAdd"
  op: "BiasAdd"
  input: "conv8_1_h/Conv2D"
  input: "conv8_1_h/bias"
}
node {
  name: "conv8_1_h/Relu"
  op: "Relu"
  input: "conv8_1_h/BiasAdd"
}
node {
  name: "conv8_2_h/Conv2D"
  op: "Conv2D"
  input: "conv8_1_h/Relu"
  input: "conv8_2_h/weights"
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
}
node {
  name: "conv8_2_h/BiasAdd"
  op: "BiasAdd"
  input: "conv8_2_h/Conv2D"
  input: "conv8_2_h/bias"
}
node {
  name: "conv8_2_h/Relu"
  op: "Relu"
  input: "conv8_2_h/BiasAdd"
}
node {
  name: "conv9_1_h/Conv2D"
  op: "Conv2D"
  input: "conv8_2_h/Relu"
  input: "conv9_1_h/weights"
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
}
node {
  name: "conv9_1_h/BiasAdd"
  op: "BiasAdd"
  input: "conv9_1_h/Conv2D"
  input: "conv9_1_h/bias"
}
node {
  name: "conv9_1_h/Relu"
  op: "Relu"
  input: "conv9_1_h/BiasAdd"
}
node {
  name: "conv9_2_h/Conv2D"
  op: "Conv2D"
  input: "conv9_1_h/Relu"
  input: "conv9_2_h/weights"
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
}
node {
  name: "conv9_2_h/BiasAdd"
  op: "BiasAdd"
  input: "conv9_2_h/Conv2D"
  input: "conv9_2_h/bias"
}
node {
  name: "conv9_2_h/Relu"
  op: "Relu"
  input: "conv9_2_h/BiasAdd"
}
node {
  name: "conv9_2_mbox_loc/Conv2D"
  op: "Conv2D"
  input: "conv9_2_h/Relu"
  input: "conv9_2_mbox_loc/weights"
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
}
node {
  name: "conv9_2_mbox_loc/BiasAdd"
  op: "BiasAdd"
  input: "conv9_2_mbox_loc/Conv2D"
  input: "conv9_2_mbox_loc/bias"
}
node {
  name: "flatten_5/Reshape"
  op: "Flatten"
  input: "conv9_2_mbox_loc/BiasAdd"
}
node {
  name: "conv9_2_mbox_conf/Conv2D"
  op: "Conv2D"
  input: "conv9_2_h/Relu"
  input: "conv9_2_mbox_conf/weights"
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
}
node {
  name: "conv9_2_mbox_conf/BiasAdd"
  op: "BiasAdd"
  input: "conv9_2_mbox_conf/Conv2D"
  input: "conv9_2_mbox_conf/bias"
}
node {
  name: "flatten_11/Reshape"
  op: "Flatten"
  input: "conv9_2_mbox_conf/BiasAdd"
}
node {
  name: "conv8_2_mbox_loc/Conv2D"
  op: "Conv2D"
  input: "conv8_2_h/Relu"
  input: "conv8_2_mbox_loc/weights"
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
}
node {
  name: "conv8_2_mbox_loc/BiasAdd"
  op: "BiasAdd"
  input: "conv8_2_mbox_loc/Conv2D"
  input: "conv8_2_mbox_loc/bias"
}
node {
  name: "flatten_4/Reshape"
  op: "Flatten"
  input: "conv8_2_mbox_loc/BiasAdd"
}
node {
  name: "conv8_2_mbox_conf/Conv2D"
  op: "Conv2D"
  input: "conv8_2_h/Relu"
  input: "conv8_2_mbox_conf/weights"
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
}
node {
  name: "conv8_2_mbox_conf/BiasAdd"
  op: "BiasAdd"
  input: "conv8_2_mbox_conf/Conv2D"
  input: "conv8_2_mbox_conf/bias"
}
node {
  name: "flatten_10/Reshape"
  op: "Flatten"
  input: "conv8_2_mbox_conf/BiasAdd"
}
node {
  name: "conv7_2_mbox_loc/Conv2D"
  op: "Conv2D"
  input: "conv7_2_h/Relu"
  input: "conv7_2_mbox_loc/weights"
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
}
node {
  name: "conv7_2_mbox_loc/BiasAdd"
  op: "BiasAdd"
  input: "conv7_2_mbox_loc/Conv2D"
  input: "conv7_2_mbox_loc/bias"
}
node {
  name: "flatten_3/Reshape"
  op: "Flatten"
  input: "conv7_2_mbox_loc/BiasAdd"
}
node {
  name: "conv7_2_mbox_conf/Conv2D"
  op: "Conv2D"
  input: "conv7_2_h/Relu"
  input: "conv7_2_mbox_conf/weights"
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
}
node {
  name: "conv7_2_mbox_conf/BiasAdd"
  op: "BiasAdd"
  input: "conv7_2_mbox_conf/Conv2D"
  input: "conv7_2_mbox_conf/bias"
}
node {
  name: "flatten_9/Reshape"
  op: "Flatten"
  input: "conv7_2_mbox_conf/BiasAdd"
}
node {
  name: "conv6_2_mbox_loc/Conv2D"
  op: "Conv2D"
  input: "conv6_2_h/Relu"
  input: "conv6_2_mbox_loc/weights"
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
}
node {
  name: "conv6_2_mbox_loc/BiasAdd"
  op: "BiasAdd"
  input: "conv6_2_mbox_loc/Conv2D"
  input: "conv6_2_mbox_loc/bias"
}
node {
  name: "flatten_2/Reshape"
  op: "Flatten"
  input: "conv6_2_mbox_loc/BiasAdd"
}
node {
  name: "conv6_2_mbox_conf/Conv2D"
  op: "Conv2D"
  input: "conv6_2_h/Relu"
  input: "conv6_2_mbox_conf/weights"
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
}
node {
  name: "conv6_2_mbox_conf/BiasAdd"
  op: "BiasAdd"
  input: "conv6_2_mbox_conf/Conv2D"
  input: "conv6_2_mbox_conf/bias"
}
node {
  name: "flatten_8/Reshape"
  op: "Flatten"
  input: "conv6_2_mbox_conf/BiasAdd"
}
node {
  name: "fc7_mbox_loc/Conv2D"
  op: "Conv2D"
  input: "last_relu"
  input: "fc7_mbox_loc/weights"
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
}
node {
  name: "fc7_mbox_loc/BiasAdd"
  op: "BiasAdd"
  input: "fc7_mbox_loc/Conv2D"
  input: "fc7_mbox_loc/bias"
}
node {
  name: "flatten_1/Reshape"
  op: "Flatten"
  input: "fc7_mbox_loc/BiasAdd"
}
node {
  name: "mbox_loc"
  op: "ConcatV2"
  input: "flatten/Reshape"
  input: "flatten_1/Reshape"
  input: "flatten_2/Reshape"
  input: "flatten_3/Reshape"
  input: "flatten_4/Reshape"
  input: "flatten_5/Reshape"
  input: "mbox_loc/axis"
}
node {
  name: "fc7_mbox_conf/Conv2D"
  op: "Conv2D"
  input: "last_relu"
  input: "fc7_mbox_conf/weights"
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
}
node {
  name: "fc7_mbox_conf/BiasAdd"
  op: "BiasAdd"
  input: "fc7_mbox_conf/Conv2D"
  input: "fc7_mbox_conf/bias"
}
node {
  name: "flatten_7/Reshape"
  op: "Flatten"
  input: "fc7_mbox_conf/BiasAdd"
}
node {
  name: "mbox_conf"
  op: "ConcatV2"
  input: "flatten_6/Reshape"
  input: "flatten_7/Reshape"
  input: "flatten_8/Reshape"
  input: "flatten_9/Reshape"
  input: "flatten_10/Reshape"
  input: "flatten_11/Reshape"
  input: "mbox_conf/axis"
}
node {
  name: "mbox_conf_reshape"
  op: "Reshape"
  input: "mbox_conf"
  input: "reshape_before_softmax"
}
node {
  name: "mbox_conf_softmax"
  op: "Softmax"
  input: "mbox_conf_reshape"
  attr {
    key: "axis"
    value {
      i: 2
    }
  }
}
node {
  name: "mbox_conf_flatten"
  op: "Flatten"
  input: "mbox_conf_softmax"
}
node {
  name: "PriorBox_0"
  op: "PriorBox"
  input: "conv4_3_norm/mul_1"
  input: "data"
  attr {
    key: "aspect_ratio"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 1
          }
        }
        float_val: 2.0
      }
    }
  }
  attr {
    key: "clip"
    value {
      b: false
    }
  }
  attr {
    key: "flip"
    value {
      b: true
    }
  }
  attr {
    key: "max_size"
    value {
      i: 60
    }
  }
  attr {
    key: "min_size"
    value {
      i: 30
    }
  }
  attr {
    key: "offset"
    value {
      f: 0.5
    }
  }
  attr {
    key: "step"
    value {
      f: 8.0
    }
  }
  attr {
    key: "variance"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 4
          }
        }
        float_val: 0.10000000149
        float_val: 0.10000000149
        float_val: 0.20000000298
        float_val: 0.20000000298
      }
    }
  }
}
node {
  name: "PriorBox_1"
  op: "PriorBox"
  input: "last_relu"
  input: "data"
  attr {
    key: "aspect_ratio"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 2
          }
        }
        float_val: 2.0
        float_val: 3.0
      }
    }
  }
  attr {
    key: "clip"
    value {
      b: false
    }
  }
  attr {
    key: "flip"
    value {
      b: true
    }
  }
  attr {
    key: "max_size"
    value {
      i: 111
    }
  }
  attr {
    key: "min_size"
    value {
      i: 60
    }
  }
  attr {
    key: "offset"
    value {
      f: 0.5
    }
  }
  attr {
    key: "step"
    value {
      f: 16.0
    }
  }
  attr {
    key: "variance"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 4
          }
        }
        float_val: 0.10000000149
        float_val: 0.10000000149
        float_val: 0.20000000298
        float_val: 0.20000000298
      }
    }
  }
}
node {
  name: "PriorBox_2"
  op: "PriorBox"
  input: "conv6_2_h/Relu"
  input: "data"
  attr {
    key: "aspect_ratio"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 2
          }
        }
        float_val: 2.0
        float_val: 3.0
      }
    }
  }
  attr {
    key: "clip"
    value {
      b: false
    }
  }
  attr {
    key: "flip"
    value {
      b: true
    }
  }
  attr {
    key: "max_size"
    value {
      i: 162
    }
  }
  attr {
    key: "min_size"
    value {
      i: 111
    }
  }
  attr {
    key: "offset"
    value {
      f: 0.5
    }
  }
  attr {
    key: "step"
    value {
      f: 32.0
    }
  }
  attr {
    key: "variance"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 4
          }
        }
        float_val: 0.10000000149
        float_val: 0.10000000149
        float_val: 0.20000000298
        float_val: 0.20000000298
      }
    }
  }
}
node {
  name: "PriorBox_3"
  op: "PriorBox"
  input: "conv7_2_h/Relu"
  input: "data"
  attr {
    key: "aspect_ratio"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 2
          }
        }
        float_val: 2.0
        float_val: 3.0
      }
    }
  }
  attr {
    key: "clip"
    value {
      b: false
    }
  }
  attr {
    key: "flip"
    value {
      b: true
    }
  }
  attr {
    key: "max_size"
    value {
      i: 213
    }
  }
  attr {
    key: "min_size"
    value {
      i: 162
    }
  }
  attr {
    key: "offset"
    value {
      f: 0.5
    }
  }
  attr {
    key: "step"
    value {
      f: 64.0
    }
  }
  attr {
    key: "variance"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 4
          }
        }
        float_val: 0.10000000149
        float_val: 0.10000000149
        float_val: 0.20000000298
        float_val: 0.20000000298
      }
    }
  }
}
node {
  name: "PriorBox_4"
  op: "PriorBox"
  input: "conv8_2_h/Relu"
  input: "data"
  attr {
    key: "aspect_ratio"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 1
          }
        }
        float_val: 2.0
      }
    }
  }
  attr {
    key: "clip"
    value {
      b: false
    }
  }
  attr {
    key: "flip"
    value {
      b: true
    }
  }
  attr {
    key: "max_size"
    value {
      i: 264
    }
  }
  attr {
    key: "min_size"
    value {
      i: 213
    }
  }
  attr {
    key: "offset"
    value {
      f: 0.5
    }
  }
  attr {
    key: "step"
    value {
      f: 100.0
    }
  }
  attr {
    key: "variance"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 4
          }
        }
        float_val: 0.10000000149
        float_val: 0.10000000149
        float_val: 0.20000000298
        float_val: 0.20000000298
      }
    }
  }
}
node {
  name: "PriorBox_5"
  op: "PriorBox"
  input: "conv9_2_h/Relu"
  input: "data"
  attr {
    key: "aspect_ratio"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 1
          }
        }
        float_val: 2.0
      }
    }
  }
  attr {
    key: "clip"
    value {
      b: false
    }
  }
  attr {
    key: "flip"
    value {
      b: true
    }
  }
  attr {
    key: "max_size"
    value {
      i: 315
    }
  }
  attr {
    key: "min_size"
    value {
      i: 264
    }
  }
  attr {
    key: "offset"
    value {
      f: 0.5
    }
  }
  attr {
    key: "step"
    value {
      f: 300.0
    }
  }
  attr {
    key: "variance"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 4
          }
        }
        float_val: 0.10000000149
        float_val: 0.10000000149
        float_val: 0.20000000298
        float_val: 0.20000000298
      }
    }
  }
}
node {
  name: "mbox_priorbox"
  op: "ConcatV2"
  input: "PriorBox_0"
  input: "PriorBox_1"
  input: "PriorBox_2"
  input: "PriorBox_3"
  input: "PriorBox_4"
  input: "PriorBox_5"
  input: "mbox_loc/axis"
}
node {
  name: "detection_out"
  op: "DetectionOutput"
  input: "mbox_loc"
  input: "mbox_conf_flatten"
  input: "mbox_priorbox"
  attr {
    key: "background_label_id"
    value {
      i: 0
    }
  }
  attr {
    key: "code_type"
    value {
      s: "CENTER_SIZE"
    }
  }
  attr {
    key: "confidence_threshold"
    value {
      f: 0.00999999977648
    }
  }
  attr {
    key: "keep_top_k"
    value {
      i: 200
    }
  }
  attr {
    key: "nms_threshold"
    value {
      f: 0.449999988079
    }
  }
  attr {
    key: "num_classes"
    value {
      i: 2
    }
  }
  attr {
    key: "share_location"
    value {
      b: true
    }
  }
  attr {
    key: "top_k"
    value {
      i: 400
    }
  }
}
node {
  name: "reshape_before_softmax"
  op: "Const"
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 3
          }
        }
        int_val: 0
        int_val: -1
        int_val: 2
      }
    }
  }
}
library {
}