# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author:johnsondiao
@File: hmm_tokenizer.py
@Time: 2022-03-25 11:39
@Desc:
"""
from tqdm import tqdm
import numpy as np
import os
import pickle


def make_label(text_str):
    """
        根据text转换为BMES的字符
    :param text_str:
    :return:
    """
    text_len = len(text_str)
    if text_len == 1:
        return "S"
    return "B" + "M" * (text_len - 2) + "E"


def text_to_state(file_path="./data/all_train_text.txt"):
    """
    将语料库的词转换为BMES的格式，再写入文件
    :param file_path:
    :return:
    """
    # 每一行当做一篇文档
    documents = open(file_path, "r", encoding="utf-8").read().split("\n")

    with open("./data/all_train_state.txt", "w", encoding="utf-8") as f:
        for index, sentence in tqdm(enumerate(documents)):
            if sentence:
                state_ = ""
                for word in sentence.split(" "):
                    if word:
                        state_ = state_ + make_label(word) + " "
                if index != len(documents) - 1:
                    state_ = state_ + "\n"
                f.write(state_)


class HMM(object):
    def __init__(self, file_text="./data/all_train_text.txt", file_state="./data/all_train_state.txt"):
        self.all_texts = open(file_text, "r", encoding="utf-8").read().split("\n")
        self.all_states = open(file_state, "r", encoding="utf-8").read().split("\n")
        self.states_to_index = {"B": 0, "M": 1, "E": 2, "S": 3}
        self.index_to_states = ["B", "M", "E", "S"]
        self.len_states = len(self.states_to_index)

        # [0. 0. 0. 0.]
        self.init_matrix = np.zeros(self.len_states)
        # [[0. 0. 0. 0.]
        #  [0. 0. 0. 0.]
        #  [0. 0. 0. 0.]
        #  [0. 0. 0. 0.]]
        self.transfer_matrix = np.zeros((self.len_states, self.len_states))
        # 发射矩阵使用2级字典嵌套, total存储当前状态出现的次数，为后面的归一化使用
        self.emit_matrix = {"B": {"total": 0}, "M": {"total": 0}, "S": {"total": 0}, "E": {"total": 0},}

    def calc_init_matrix(self, state):
        """计算初识概率矩阵"""
        self.init_matrix[self.states_to_index[state[0]]] += 1

    def calc_transfer_matrix(self, states):
        """转移概率矩阵"""
        states = "".join(states)
        state1 = states[:-1]
        state2 = states[1:]
        for s1, s2 in zip(state1, state2):
            self.transfer_matrix[self.states_to_index[s1], self.states_to_index[s2]] += 1

    def calc_emit_matrix(self, words, states):
        """发射矩阵"""
        for word, state in zip("".join(words), "".join(states)):
            if word not in self.emit_matrix:
                self.emit_matrix[word] = {"total": 0}
            self.emit_matrix[word][state] = self.emit_matrix[word].get(state, 0) + 1
            self.emit_matrix[word]["total"] += 1

    def normalize(self):
        self.init_matrix = self.init_matrix / np.sum(self.init_matrix)
        self.transfer_matrix = self.transfer_matrix / np.sum(self.transfer_matrix, axis=1, keepdims=True)
        self.emit_matrix = {word: {state: time / states["total"] for state, time in states.items() if state != "total"}
                            for word, states in self.emit_matrix.items()}

    def train(self):
        if os.path.exists("./data/hmm.pkl"):
            self.init_matrix, self.transfer_matrix, self.emit_matrix = pickle.load(open("./data/hmm.pkl", "rb"))
            return

        for words, states in tqdm(zip(self.all_texts, self.all_states)):
            words = words.split(" ")
            states = states.split(" ")
            self.calc_init_matrix(states[0])
            self.calc_transfer_matrix(states)
            self.calc_emit_matrix(words, states)
        self.normalize()
        pickle.dump([self.init_matrix, self.transfer_matrix, self.emit_matrix], open("./data/hmm.pkl", "wb"))


if __name__ == '__main__':
    # text_to_state()

    hmm = HMM()
    hmm.train()


