# 1) VGG-16 (파라미터는 많으나, 생각보다 빠르고 성능이 좋음)
# 2) ResNet50-SE (ResNet101이나 ResNet152 대비 efficient, GPU 서빙의 마지노선)
# 3) ResNeXT101-SE + FPN (FPN은 detection등에 적용할 때 성능에 큰 영향)
# 4) Xception 계통 (의외로 효율적이고 강력한 모델, group-conv 기반)
# 5) EfficientNet-B4 (효율적이고 성능이 좋지만, dw-conv 기반)
# + AmoebaNet등의 NAS 기반 거대 모델들은 complexity나 scalability 이슈로 추천하지 않습니다
# 1) ResNet-18 (ResNet-50보다 빠르지만, 여전히 크기가 큼)
# 2) Xception 계통 (CPU용, Xception을 작게 만들어 사용)
# 3) MobileNetV1 (CPU용, MobileNetV2보다 더 좋을 때가 많음)