import tensorflow as tf

#VGG16
model_vgg16 = tf.keras.applications.VGG16(weights='imagenet', include_top=False, pooling='avg')
model_vgg16_save_path = 'vgg16_model.h5'
model_vgg16.save(model_vgg16_save_path)

#ResNet50
model_resnet = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, pooling='avg')
model_resnet_save_path = 'resnet50_model.h5'
model_resnet.save(model_resnet_save_path)

#MobileNetV2
model_mobilenet = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
model_mobilenet_save_path = 'mobilenetv2_model.h5'
model_mobilenet.save(model_mobilenet_save_path)