'''
변수 초기화는 모델의 다른 연산을 실행하기 전에 반드시 명시적으로 실행해야 합니다. 가장 쉬운 방법은 모든 변수를 초기화 하는 연산을 모델 사용 전에 실행하는 것입니다.
다른 방법으로는 체크포인트 파일에서 변수 값을 복원할 수 있습니다. 다음 챕터에서 다룰 것입니다.
변수 초기화를 위한 작업(op)을 추가하기 위해 tf.global_variables_initializer()를 사용해봅시다. 모델을 모두 만들고 세션에 올린 후 이 작업을 실행할 수 있습니다.
'''
import tensorflow as tf

sess = tf.Session()

x = tf.Variable([2], dtype = tf.float32, name = 'test') #변수지정 
#변수 초기화 시켜줘야함
init = tf.global_variables_initializer()

sess.run(init) 

print(sess.run(x)) #[2.]