import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

class CnnNet(nn.Module):
    def __init__(self) -> None:
        super(CnnNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 12)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x) # 全连接层中间的dropout层
        x = torch.relu(self.fc2(x))
        x = self.dropout(x) # 全连接层中间的dropout层
        x = self.fc3(x)
        return x
    
    def train(self):
        best_acc = 0
        loss_threshold = 5 # 连续5次训练集损失没有下降就停止训练
        prev_loss = float('inf')
        no_improvement_count = 0

        for epoch in range(50):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 100 == 99:
                    print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss/100))
                    if running_loss >= prev_loss:
                        no_improvement_count += 1
                    else:
                        no_improvement_count = 0
                    prev_loss = running_loss
                    running_loss = 0.0
            
            # if no_improvement_count >= loss_threshold:
            #     print('Training stopped early due to no improvement in %d epochs.' % loss_threshold)
            #     return
            
            correct = 0
            total = 0
            with torch.no_grad():
                for data in test_loader:
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)

                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            acc = 100 * correct / total
            print('Accuracy on the test set: %d %%' % acc)

            if acc > 99.0:
                print('Accuracy reached the threshold, training stopped.')
                return
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), './model_params.pt')

    def predict(self):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(correct / total) 

    
transform = transforms.Compose(
    [transforms.Grayscale(),
     transforms.ToTensor(),
     transforms.Normalize((0.5), (0.5))]
)

trainset = torchvision.datasets.ImageFolder(root = './train2', transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)

testset = torchvision.datasets.ImageFolder(root='/Volumes/LCZ/test_data', transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CnnNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001) # L2正则化系数

# model.train()
model.load_state_dict(torch.load('./model_params.pt'))
model.predict()