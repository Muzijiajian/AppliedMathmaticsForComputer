#!/usr/bin/env sh
MY=/home/lord/caffe-master/ldj_workspace/caltech101/myfile

echo "Create train lmdb.."
rm -rf $MY/image_train_lmdb
build/tools/convert_imageset \
--shuffle \
--resize_width=256 \
--resize_height=256 \
/home/lord/caffe-master/data/caltech101/101_ObjectCategories/ \
$MY/train2.txt \
$MY/image_train_lmdb

echo "Create test lmdb.."
rm -rf $MY/image_test_lmdb
build/tools/convert_imageset \
--shuffle \
--resize_width=256 \
--resize_height=256 \
/home/lord/caffe-master/data/caltech101/101_ObjectCategories/ \
$MY/test2.txt \
$MY/image_test_lmdb

echo "All done"

# 10行通过两个目录拼接给出一个完整的图片路径
# 11行用于指定包含所有图片文件的目录，格式为 category/目录1/image1