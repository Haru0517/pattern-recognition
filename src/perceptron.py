"""
1316190104 近澤悠登
"""

from matplotlib import pyplot as plt
import numpy as np
import random


def learning(_x_list: list, _rate: float):
    """x_listを学習して，学習過程の(w0, w1)を返す．

    Args:
        _x_list (list): 学習用データ．
        _rate (float): ρ．

    Returns:
        ret_w_list (list): １ループごとの(w0, w1)の座標．
    """
    # 適当にw0とw1を選択
    w0 = random.randint(-10, 10)
    w1 = random.randint(0, 10)
    w = np.array([w0, w1])
    print(w)

    ret_w_list = [w]

    # 全パターン識別できるまでループ
    while True:
        error = False
        for x, correct_class in _x_list:
            # 拡張特徴ベクトルにする
            x = np.array([1, x])

            # g(x)を計算
            score = np.dot(w, x)

            # 誤りがあれば重みを修正
            if score <= 0 and correct_class == 1:
                w = w + x * _rate
                error = True
            elif score >= 0 and correct_class == 2:
                w = w - x * _rate
                error = True

        # 誤りがなければ終了
        if error is False:
            break

        ret_w_list.append(w)

    return ret_w_list


def draw_graph(_x_list: list, _w_list: list, _rate: float):
    """グラフを描画する．

    Args:
        _x_list (list): 学習用データ．
        _w_list (list): 学習過程の重みリスト．
        _rate (float): ρ．

    Returns:

    """
    x_range = np.array([-1, 15])
    y_range = np.array([-10, 10])

    # 線をプロット
    for feature, _ in _x_list:
        tilt = -1 * feature  # g(x)=w0+w1*x=0 → w0/w1=-x
        plt.plot(x_range, x_range * tilt, c='green')

    # 点をプロット
    for i, w in enumerate(_w_list):
        plt.plot(w[1], w[0], marker=f'$•{i+1}$', markersize=10)

    # 解領域を塗りつぶし
    y1 = -1*x_range*_x_list[2][0]
    y2 = -1*x_range*_x_list[3][0]
    plt.fill_between(x_range, y1, y2, facecolor='y', alpha=0.5)

    # その他グラフの設定
    plt.axvline(x=0, c='black')
    plt.axhline(y=0, c='black')
    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.xlabel('w1', fontsize=10)
    plt.ylabel('w0', fontsize=10)
    plt.title(f"ρ={_rate}")
    plt.show()


if __name__ == '__main__':
    # 1.2, 2.0, 3.6
    rate = 1.2
    x_list = [(-1.5, 2), (-1.0, 2), (-0.5, 2), (-0.2, 1), (0.2, 1), (1.2, 1)]
    w_list = learning(x_list, rate)
    draw_graph(x_list, w_list, rate)
