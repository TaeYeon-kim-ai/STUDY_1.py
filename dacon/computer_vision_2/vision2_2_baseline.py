import os
from typing import Tuple, Sequence, Callable
import csv
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.optim as optim
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary

from torchvision import transforms
from torchvision.models import resnet50

class MnistDataset(Dataset):
    def __init__(
        self,
        dir: os.PathLike,
        image_ids: os.PathLike,
        transforms: Sequence[Callable]
    ) -> None:
        self.dir = dir
        self.transforms = transforms

        self.labels = {}
        with open(image_ids, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                self.labels[int(row[0])] = list(map(int, row[1:]))

        self.image_ids = list(self.labels.keys())

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> Tuple[Tensor]:
        image_id = self.image_ids[index]
        image = Image.open(
            os.path.join(
                self.dir, f'{str(image_id).zfill(5)}.png')).convert('RGB')
        target = np.array(self.labels.get(image_id)).astype(np.float32)

        if self.transforms is not None:
            image = self.transforms(image)

        return image, target


transforms_train = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

transforms_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


trainset = MnistDataset("C:/data/vision_2/dirty_mnist_2nd_noise_clean","C:/data/vision_2/dirty_mnist_2nd_answer.csv", transforms_train)
testset = MnistDataset("C:/data/vision_2/test_dirty_mnist_2nd_noise_clean","C:/data/vision_2/sample_submission.csv", transforms_test)

train_loader = DataLoader(trainset, batch_size=256, num_workers=8)
test_loader = DataLoader(testset, batch_size=32, num_workers=4)



class MnistModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resnet = resnet50(pretrained=True)
        self.classifier = nn.Linear(1000, 26)

    def forward(self, x):
        x = self.resnet(x)
        x = self.classifier(x)

        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MnistModel().to(device)
print(summary(model, input_size=(1, 3, 256, 256), verbose=0))

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MultiLabelSoftMarginLoss()

num_epochs = 10
model.train()

for epoch in range(num_epochs):
    for i, (images, targets) in enumerate(train_loader):
        optimizer.zero_grad()

        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            outputs = outputs > 0.5
            acc = (outputs == targets).float().mean()
            print(f'{epoch}: {loss.item():.5f}, {acc.item():.5f}')

if __name__ == '__main__':
    
    # 평가 폴더를 열음
    submit = pd.read_csv("C:/data/vision_2/sample_submission.csv")

    # 이벨류 모드로 전환
    model.eval()

    # 베치사이즈는 테스트로더 베치사이즈
    batch_size = test_loader.batch_size
    # 인덱스 0부터 시작
    batch_index = 0
    # 이벨류 모드를 테스트 셋으로 진행하고 파일에 입력함
    for i, (images, targets) in enumerate(test_loader):
        images = images.to(device)
        targets = targets.to(device)
        outputs = model(images)
        outputs = outputs > 0.5
        batch_index = i * batch_size
        submit.iloc[batch_index:batch_index+batch_size, 1:] = \
            outputs.long().squeeze(0).detach().cpu().numpy()

    # 저장함
    submit.to_csv("C:/data/vision_2/submission.csv", index=False)

    del images
    del targets

print("제발 돌아가라 ㅜㅜ 잠좀 자자!!!!!")
print("제발 돌아가라 ㅜㅜ 잠좀 자자!!!!!")
print("제발 돌아가라 ㅜㅜ 잠좀 자자!!!!!")
print("제발 돌아가라 ㅜㅜ 잠좀 자자!!!!!")
print("제발 돌아가라 ㅜㅜ 잠좀 자자!!!!!")

