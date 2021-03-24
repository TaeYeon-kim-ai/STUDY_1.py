from tensorflow.train import latest_checkpoint, restore


#tf.train.latest_checkpoint하여 최신 체크 포인트 파일을 가져온 다음 다음을 사용하여 수동으로로드 할 수 있습니다 ckpt.restore.

checkpoint_path = "./checkpoints/train"
ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,

ckpt_path = tf.train.latest_checkpoint(checkpoint_path)
ckpt.restore(ckpt_path)



#=========================================

from tensorflow.train import checkpoint, checkpoint_path
#Checkpoint객체는 체크 포인트 파일에 추적 가능한 객체의 단일 또는 그룹 중 하나를 저장하는 구성 할 수있습니다.
#save_counter번호 매기기 체크 포인트를 유지합니다 .

#기본형
tf.train.Checkpoint(
    root=None, **kwargs #	The root object to checkpoint. #	Keyword arguments are set as attributes of this object, and are saved with the checkpoint. Values must be trackable objects.
)

#예시
model = tf.keras.Model(...)
checkpoint = tf.train.Checkpoint(model)

# Save a checkpoint to /tmp/training_checkpoints-{save_counter}. Every time 매번 저장
# checkpoint.save is called, the save counter is increased. #checkpoint.save가 호출되면 저장 카운터가 증가합니다.
save_path = checkpoint.save('/tmp/training_checkpoints')

# Restore the checkpointed values to the `model` object. # 체크 포인트 값을`model` 객체로 복원합니다.
checkpoint.restore(save_path)

#save_counter	Incremented when save() is called. Used to number checkpoints.


class Regress(tf.keras.Model):
    
  def __init__(self):
    super(Regress, self).__init__()
    self.input_transform = tf.keras.layers.Dense(10)
    # ...

  def call(self, inputs):
    x = self.input_transform(inputs)
    # ...
#레이어 Model에 "input_transform"이라는 종속성 이 있으며 Dense, 이는 변수에 따라 달라집니다. 
# 결과적으로 Regressusing 인스턴스를 tf.train.Checkpoint저장하면 Dense레이어에서 생성 된 모든 변수도 저장됩니다 .
# 여러 작업자에게 변수가 할당되면 각 작업자는 자체 검사 점 섹션을 작성합니다. 
# 이러한 섹션은 병합 / 재 인덱싱되어 단일 체크 포인트로 작동합니다. 
# 이렇게하면 모든 변수를 하나의 작업자에 복사하는 것을 방지 할 수 있지만 모든 작업자가 공통 파일 시스템을 볼 수 있어야합니다.

# 이 함수는 Keras Model save_weights 함수.tf.케라스와 약간 다릅니다.
# model.save_weights는 tf.train과 파일 경로에 지정된 이름의 체크포인트 파일을 생성합니다.
# 검사점 파일 이름의 접두사로 파일 경로를 사용하여 검사점 번호를 지정합니다. 
# 이것 말고도 모델.save_buffer 및 tf.train체크포인트(모델).저장()은 동일합니다.