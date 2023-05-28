import sys
from sklearn_crfsuite import CRF

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
    
    def write(self, words_test, labels_pred):
        # 将预测结果写入文件
        with open('output_'+ self.language +'_for_PJ2_2_test.txt', 'w', encoding='utf-8') as f:
            for i in range(len(words_test)):
                for j in range(len(words_test[i])):
                    f.write(words_test[i][j] + ' ' + labels_pred[i][j] + '\n')
                f.write('\n')

class CRF_model:
    def __init__(self, words_train, labels_train, words_test):
        self.words_train = [self.sent2features(s) for s in words_train]
        self.labels_train = [self.sent2labels(s) for s in labels_train]
        self.words_test = [self.sent2features(s) for s in words_test]

    # 将数据转化为CRF模块需要的特征格式
    def word2features(self, sent, i):
        # sent是一个句子，i是句子中的第i个词
        word = sent[i]
        features = {
            'bias': 1.0,
            'U00': word[-2] if len(word)>=2 else '',
            'U01': word[-1] if len(word)>=1 else '',
            'U02': word[0] if len(word)>=1 else '',
            'U03': word[1] if len(word)>=2 else '',
            'U04': word[2] if len(word)>=3 else '',
            'U05': word[-2:] if len(word)>=2 else '',
            'U06': word[-1:] + word[0] if len(word)>=2 else '',
            'U07': word[-1:] + word[1] if len(word)>=2 else '',
            'U08': word[0] + word[1] if len(word)>=2 else '',
            'U09': word[1:] if len(word)>=2 else ''
        }
        # 如果不是句子的第一个词
        # 那么就添加上一个词的特征
        # 否则添加一个空字符串
        # 以此类推
        if i > 0:
            # word1是上一个词
            word1 = sent[i-1][0]
            features.update({
                'B00': word1[-2] if len(word1)>=2 else '',
                'B01': word1[-1] if len(word1)>=1 else '',
                'B02': word1[0] if len(word1)>=1 else '',
                'B03': word1[1] if len(word1)>=2 else '',
                'B04': word1[2] if len(word1)>=3 else '',
                'B05': word1[-2:] + word[-2:] if len(word)>=2 and len(word1)>=2 else '',
                'B06': word1[-1:] + word[0] + word[-1:] + word[1:] if len(word)>=2 else word1+word,
                'B07': word1[-1:] + word[1:] + word[-1:] + word[1:] if len(word)>=2 else word1+word,
                'B08': word1[0] + word[1] + word[0] + word[1:] if len(word)>=2 else word1+word,
                'B09': word1[1:] + word[1:] if len(word)>=2 else word1+word
            })
        # 如果是句子的第一个词
        else:
            # 添加BOS特征
            features['BOS'] = True
        # 如果不是句子的最后一个词
        if i < len(sent)-1:
            word1 = sent[i+1][0]
            features.update({
                'A00': word1[-2] if len(word1)>=2 else '',
                'A01': word1[-1] if len(word1)>=1 else '',
                'A02': word1[0] if len(word1)>=1 else '',
                'A03': word1[1] if len(word1)>=2 else '',
                'A04': word1[2] if len(word1)>=3 else '',
                'A05': word[-2:] + word1[-2:] if len(word)>=2 and len(word1)>=2 else '',
                'A06': word[-1:] + word[0] + word1[-1:] + word1[1:] if len(word)>=2 else word+word1,
                'A07': word[-1:] + word1[1:] + word1[-1:] + word1[1:] if len(word)>=2 else word+word1,
                'A08': word[0] + word[1] + word1[0] + word1[1:] if len(word)>=2 else word+word1,
                'A09': word[1:] + word1[1:] if len(word)>=2 else word+word1
            })
        # 如果是句子的最后一个词
        else:
            features['EOS'] = True

        return features

    def sent2features(self, sent):
        return [self.word2features(sent, i) for i in range(len(sent))]

    def sent2labels(self, sent):
        return [label for label in sent]
    
    def fit(self, algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True):
        # 训练CRF模型
        # algorithm: 训练算法, c1: L1正则化系数, c2: L2正则化系数, max_iterations: 最大迭代次数, all_possible_transitions: 是否允许所有可能的转移
        self.crf = CRF(algorithm=algorithm, c1=c1, c2=c2, max_iterations=max_iterations, all_possible_transitions=all_possible_transitions, verbose=True)
        self.crf.fit(self.words_train, self.labels_train)

    def predict(self):
        # 预测
        return self.crf.predict(self.words_test)

def main():
    # 检查命令行参数
    if len(sys.argv) != 4:
        print('Usage: python PJ2_2.py <train_file> <test_file> <language>')
        return   
    # 读取命令行参数
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    language = sys.argv[3]
    # 读取数据
    pre = preprocess(train_file, test_file, language)
    words_train, labels_train, words_test= pre.read()

    print('Data reading')
    crf = CRF_model(words_train, labels_train, words_test)
    print('Data read completed')

    print('Model fitting')
    crf.fit()

    print('Model fit completed')

    print('Model predicting')
    labels_pred = crf.predict()
    pre.write(words_test, labels_pred)
    print('Model predict completed')

if __name__ == '__main__':
    # python NER/PJ2_2.py NER/English/train.txt NER/English/validation.txt english
    # python NER/PJ2_2.py NER/Chinese/train.txt NER/Chinese/validation.txt chinese
    main()