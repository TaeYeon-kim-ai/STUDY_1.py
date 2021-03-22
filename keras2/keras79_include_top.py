from tensorflow.keras.applications import VGG16

#model = VGG16(weights = 'imagenet', include_top=False, input_shape = (224, 224, 3))
model = VGG16() #하단에 Dense부분 flatten 추가됨 커스터 마이징 가능

model.trainable = False
model.summary()
print(len(model.weights))
print(len(model.trainable_weights))

# vgg16(include_top=False)
# =================================================================
# Total params: 14,714,688
# Trainable params: 0
# Non-trainable params: 14,714,688
# _________________________________________________________________p
# 26
# 0

#vgg16() = include_top = True
# =================================================================
# Total params: 138,357,544
# Trainable params: 0
# Non-trainable params: 138,357,544
# _________________________________________________________________
# 32
# 0






