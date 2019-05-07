# -*- coding: utf-8 -*-

import argparse
import random
import numpy as np

def learning(x_list, larning_rate):
    # wの初期化
    w0 = random.randint(-10, 10)
    w1 = random.randint(-10, 10)
    w = np.asarray([w0, w1])
    print("Learning start :\tw0 = %.2f,\tw1 = %.2f" % (w[0], w[1]))

    iteration = 0
    number_of_errors = 1

    while not number_of_errors == 0:
        # x_listをシャッフルする
        np.random.shuffle(x_list)
        number_of_errors = 0

        # xを一つずつ取り出す
        for x, correct_class in x_list:
            # xを拡張ベクトルにする
            x = np.asarray([1, x])

            # 内積をとる
            score = np.dot(w, x)

            # 分類が誤っていれば重みを修正する
            if score <= 0 and correct_class == 1:
                w = w + x * larning_rate
                number_of_errors += 1
            elif score >= 0 and correct_class == 2:
                w = w - x * larning_rate
                number_of_errors += 1

        # iterationごとに結果を表示
        iteration += 1
        print("Iteration %d :\t\tw0 = %.2f,\tw1 = %.2f" % (iteration, w[0], w[1]))

    print("Learning end :\t\tw0 = %.2f,\tw1 = %.2f" % (w[0], w[1]))
    print("x = %f" % -(w[0]/w[1]))

if __name__ == '__main__':
    # 引数の設定
    ap = argparse.ArgumentParser()
    ap.add_argument("learning_late", default=1, help="required : please set learning rate.", type=float)
    args = ap.parse_args()

    # 引数を変数に格納
    learning_rate = args.learning_late

    # データ(x, class)
    x_list = np.asarray([(-1.5, 2), (-1.0, 2), (-0.5, 2), (-0.2, 1), (0.2, 1), (1.2, 1)])

    # run
    learning(x_list, learning_rate)
