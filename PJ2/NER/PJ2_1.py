import sys
import numpy as np

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

        if self.language.lower() == 'chinese':
            # states = sorted(list(set(Y_train)))
            # 如上方法获得的state只有28种，但是实际上有33种
            states = ['B-NAME', 'M-NAME', 'E-NAME', 'S-NAME',
                    'B-CONT', 'M-CONT', 'E-CONT', 'S-CONT',
                    'B-EDU', 'M-EDU', 'E-EDU', 'S-EDU',
                    'B-TITLE', 'M-TITLE', 'E-TITLE', 'S-TITLE',
                    'B-ORG', 'M-ORG', 'E-ORG', 'S-ORG',
                    'B-RACE', 'M-RACE', 'E-RACE', 'S-RACE',
                    'B-PRO', 'M-PRO', 'E-PRO', 'S-PRO',
                    'B-LOC', 'M-LOC', 'E-LOC', 'S-LOC', 
                    'O']
        elif self.language.lower() == 'english':
            states = ['B-PER', 'I-PER',
                    'B-ORG', 'I-ORG', 
                    'B-LOC', 'I-LOC', 
                    'B-MISC', 'I-MISC',
                    'O']
        else:
            print('language must be chinese or english')
            return
        observations = sorted(list(set(words)))
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

        return words_train, labels_train, words_test, states, observations
    
    def write(self, words_test, labels_pred):
        # 将预测结果写入文件
        with open('output_'+ self.language +'_for_PJ2_1_test.txt', 'w', encoding='utf-8') as f:
            for i in range(len(words_test)):
                for j in range(len(words_test[i])):
                    f.write(words_test[i][j] + ' ' + labels_pred[i][j] + '\n')
                f.write('\n')

class HMM:
    def __init__(self, states, observations):
        # states: 状态集合
        # observations: 观测集合
        # initial_prob: 初始概率
        # transition_prob: 转移概率
        # emission_prob: 发射概率
        # unknown_word_emission_prob: 未知单词的发射概率
        self.states = states
        self.observations = observations
        self.initial_prob = np.zeros(len(states))
        self.transition_prob = np.zeros((len(states), len(states)))
        self.emission_prob = np.zeros((len(states), len(observations)))
        self.unknown_word_emission_prob = np.zeros(len(states))
    
    def train(self, words, labels):
        k = 1
        # 计算初始概率、转移概率、发射概率
        for label in labels:
            self.initial_prob[self.states.index(label[0])] += 1
            for i in range(1, len(label)):
                curr_state = self.states.index(label[i])
                prev_state = self.states.index(label[i-1])
                self.transition_prob[prev_state][curr_state] += 1
        for word, label in zip(words, labels):
            for i in range(len(word)):
                state = self.states.index(label[i])
                obs = self.observations.index(word[i])
                self.emission_prob[state][obs] += 1
        # 平滑 + 归一化
        self.initial_prob = (self.initial_prob + k) / (np.sum(self.initial_prob) + k * len(self.states))
        self.transition_prob = (self.transition_prob + k) / (np.sum(self.transition_prob, axis=1, keepdims=True) + k * len(self.states))
        self.emission_prob = (self.emission_prob + k) / (np.sum(self.emission_prob, axis=1, keepdims=True) + k * (len(self.observations) + 1))
        self.unknown_word_emission_prob = np.ones(len(self.states)) / (np.sum(self.emission_prob, axis=1) + 1)  # 未知单词的发射概率等于均匀分布
    
    
    def decode(self, words):
        result = []
        # 计算delta和phi
        # delta: 每个时刻每个状态的最大概率
        # phi: 每个时刻每个状态的最大概率对应的前一个状态
        # path: 最优路径
        for word in words:
            delta = np.zeros((len(self.states), len(word)))
            phi = np.zeros((len(self.states), len(word)), dtype=int)
            if word[0] not in self.observations:
                delta[:, 0] = self.initial_prob * self.unknown_word_emission_prob
            else:
                obs = self.observations.index(word[0])
                delta[:, 0] = self.initial_prob * self.emission_prob[:, obs]
            for t in range(1, len(word)):
                for s in range(len(self.states)):
                    if word[t] not in self.observations:
                        p = self.transition_prob[:, s] * delta[:, t-1] * self.unknown_word_emission_prob
                    else:
                        obs = self.observations.index(word[t])
                        p = self.transition_prob[:, s] * delta[:, t-1] * self.emission_prob[s, obs]
                    delta[s, t] = np.max(p)
                    phi[s, t] = np.argmax(p)
            path = [np.argmax(delta[:, -1])]
            for t in range(len(word)-1, 0, -1):
                path.append(phi[path[-1], t])
            path.reverse()
            result.append([self.states[i] for i in path])
        return result

def main():
    # 检查命令行参数
    if len(sys.argv) != 4:
        print('Usage: python PJ2_1.py <train_file> <test_file> <language>')
        return 
    # 读取命令行参数
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    language = sys.argv[3]
    # 读取数据
    pre = preprocess(train_file, test_file, language)
    words_train, labels_train, words_test, states, observations = pre.read()
    # 训练HMM模型
    hmm = HMM(states, observations)
    print('training...')
    hmm.train(words_train, labels_train)
    print('training finished')
    # 预测测试数据标签
    print('predicting...')
    labels_pred = hmm.decode(words_test)
    print('predicting finished')
    # 写入文件
    pre.write(words_test, labels_pred)

if __name__ == '__main__':
    # python NER/PJ2_1.py NER/English/train.txt NER/English/validation.txt english
    # python NER/PJ2_1.py NER/Chinese/train.txt NER/Chinese/validation.txt chinese
    main()

