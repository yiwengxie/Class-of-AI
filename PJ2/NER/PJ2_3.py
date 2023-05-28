import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchcrf import CRF

class preprocess:
    def __init__(self, train_file, test_file, language):
        self.train_file = train_file
        self.test_file = test_file
        self.language = language

    def read(self):
        # 读取训练数据
        with open(self.train_file, 'r', encoding='utf-8') as f:
            train_data = f.readlines()
        # 处理训练数据
        words_train, labels_train = [[]], [[]]
        words = []
        for line in train_data:
            if line != '\n':
                word, label = line.strip().split()
                words_train[-1].append(word)
                labels_train[-1].append(label)
                words.append(word)
            else:
                words_train.append([])
                labels_train.append([])
        while [] in words_train:
            words_train.remove([])
        while [] in labels_train:
            labels_train.remove([])
        # 读取测试数据
        with open(self.test_file, 'r', encoding='utf-8') as f:
            test_data = f.readlines()
        # 处理测试数据
        words_test = [[]]
        for line in test_data:
            if(line != '\n'):
                word, label = line.strip().split()
                words_test[-1].append(word)
            else:
                words_test.append([])
        while [] in words_test:
            words_test.remove([])

        return words_train, labels_train, words_test
    
    def toidx(self, words_train, labels_train, words_test):
        # 创建单词和标签的索引
        word2idx, tag2idx = {}, {}
        for sentence, tags in zip(words_train, labels_train):
            for word, label in zip(sentence, tags):
                word2idx.setdefault(word, len(word2idx))
                tag2idx.setdefault(label, len(tag2idx))
        word2idx.setdefault('UNK', len(word2idx))
              
        # 将句子和标签转换为索引序列
        sentences_idx = [[word2idx[word] for word in sentence] for sentence in words_train]
        test_idx = [[word2idx.get(word, word2idx['UNK']) for word in sentence] for sentence in words_test]
        labels_idx = [[tag2idx[tag] for tag in tags] for tags in labels_train]
        return word2idx, tag2idx, sentences_idx, labels_idx, test_idx
    
# 定义模型
class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag2idx, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()       
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)        
        # LSTM层
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)        
        # 线性层
        self.linear = nn.Linear(hidden_dim, len(tag2idx))      
        # CRF层
        self.crf = CRF(len(tag2idx))
        
    def forward(self, x, y):
        embedded = self.embedding(x)        
        lstm_out, _ = self.lstm(embedded)        
        emissions = self.linear(lstm_out)
        emissions = emissions.unsqueeze(0)  
        y = y.unsqueeze(0)  # add a new batch dimension     
        loss = -self.crf(emissions, y)        
        return loss

    
    def decode(self, x):
        embedded = self.embedding(x)       
        lstm_out, _ = self.lstm(embedded)       
        emissions = self.linear(lstm_out)
        emissions = emissions.unsqueeze(0)  # add a new batch dimension
        tags = self.crf.decode(emissions)
        return tags
    
class Trainer:
    def __init__(self, model, optimizer, criterion, language):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.language = language
        
    def train(self, sentences_idx, labels_idx, epochs):
        # 训练模型
        for epoch in range(epochs):
            epoch_loss = 0.0
            for sentence, tags in zip(sentences_idx, labels_idx):
                # 将句子和标签转换为PyTorch张量
                sentence = torch.tensor(sentence, dtype=torch.long)
                tags = torch.tensor(tags, dtype=torch.long)      
                # 清除之前的梯度
                self.model.zero_grad()
                # 获取模型的负对数损失
                loss = self.model(sentence, tags)
                # 反向传播并更新参数
                loss.backward()
                self.optimizer.step()
                # 累计损失
                epoch_loss += loss.item()
                
            # 打印每个训练周期的损失
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
            
    def test(self, test_idx, word2idx, tag2idx):
        # 在测试数据集上评估模型
        with torch.no_grad():
            # 将预测结果写入文件
            with open('output_'+ self.language +'_for_PJ2_3_test.txt', 'w', encoding='utf-8') as f:
                for sentence in test_idx:
                    # 将句子转换为PyTorch张量
                    sentence = torch.tensor(sentence, dtype=torch.long)
                    # 获取模型的预测标签序列
                    pred_tags = self.model.decode(sentence)
                    flat_list = [item for sublist in pred_tags for item in sublist]
                    # 将预测标签序列转换为标签
                    pred_tags = [list(tag2idx.keys())[list(tag2idx.values()).index(tag)] for tag in flat_list]
                    # 将句子转换为单词
                    words = [list(word2idx.keys())[list(word2idx.values()).index(word)] for word in sentence]
                    # 打印句子和预测标签序列
                    for i in range(len(words)):
                        f.write(words[i] + ' ' + pred_tags[i] + '\n')
                    f.write('\n')

def main():
    # 检查命令行参数
    if len(sys.argv) != 4:
        print('Usage: python PJ2_3.py <train_file> <test_file> <language>')
        return 
    # 读取命令行参数
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    language = sys.argv[3]
    # 读取数据
    print('Reading data...')
    pre = preprocess(train_file, test_file, language)
    words_train, labels_train, words_test= pre.read()
    word2idx, tag2idx, sentences_idx, labels_idx, test_idx = pre.toidx(words_train, labels_train, words_test)
    print('Reading Done!')
    # 定义超参数
    EMBEDDING_DIM = 32
    HIDDEN_DIM = 64
    EPOCHS = 20
    # 创建模型
    model = BiLSTM_CRF(len(word2idx), tag2idx, EMBEDDING_DIM, HIDDEN_DIM)
    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss() 
    # 创建训练器
    trainer = Trainer(model, optimizer, criterion, language)
    # 训练模型
    print('Training model...')
    trainer.train(sentences_idx, labels_idx, EPOCHS)
    print('Training Done!')
    # 在测试数据集上评估模型
    print('Testing model...')
    trainer.test(test_idx, word2idx, tag2idx)
    print('Testing Done!')


if __name__ == '__main__':
    # python NER/PJ2_3.py NER/English/train.txt NER/English/validation.txt english
    # python NER/PJ2_3.py NER/Chinese/train.txt NER/Chinese/validation.txt chinese
    main()
