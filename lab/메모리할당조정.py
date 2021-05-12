# 모델이 돌아가기 전에 터지는것을 방지하기 위한 코드임
# gpu메모리 할당 조절하여 터지는것 방지.
# 하지만 그만큼 훈련속도가 느려진다.

configuration = tf.compat.v1.ConfigProto()
configuration.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=configuration)