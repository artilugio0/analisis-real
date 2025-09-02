import matplotlib.pyplot as plt
import numpy as np

def plot_lim(f, a, x_min=0, x_max=10, y_min=None, y_max=None):
    x = np.linspace(x_min, x_max, 200)
    y = f(x)

    if y_min is None:
        y_min = np.min(y)
    if y_max is None:
        y_max = np.max(y)

    fig, ax = plt.subplots()

    ax.grid(visible=True, color='#DDDDDD', linestyle='--')
    ax.legend()

    ax.spines.top.set_visible(False)
    ax.spines.right.set_visible(False)
    ax.spines.bottom.set_color('#CCCCCC')
    ax.spines.left.set_color('#CCCCCC')

    ax.set_xlim(x_min, x_max)

    x_amp = abs(x_max-x_min)
    y_amp = abs(y_max-y_min)
    y_margin = y_amp*0.03
    x_arrow_margin = y_margin*0.7
    x_arrow_len = x_amp*0.06

    ax.set_ylim(y_min - y_margin, y_max + y_margin)

    ax.plot(x, y, color='blue', label='y = f(x)')

    ax.annotate("", xytext=(a-x_arrow_len, y_min - x_arrow_margin), xy=(a, y_min - x_arrow_margin), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xytext=(a+x_arrow_len, y_min - x_arrow_margin), xy=(a, y_min - x_arrow_margin), arrowprops=dict(arrowstyle="->"))

    delta = 0.00001
    left_limit = f(np.array([a-delta]))[0]
    right_limit = f(np.array([a+delta]))[0]

    # tg_line: y = dy/dx * (x - a) + left_limit
    # y_tg_line = dy_dx_left * (x-a) + left_limit

    dy_dx_left = (f(a) - f(a-delta))/(delta)
    dy_dx_right = (f(a) - f(a+delta))/(-delta)

    left_angle = np.atan(dy_dx_left) + np.pi
    left_dx = x_arrow_len * np.cos(left_angle)
    left_dy = x_arrow_len * np.sin(left_angle)

    right_angle = np.atan(dy_dx_right)
    right_dx = x_arrow_len * np.cos(right_angle)
    right_dy = x_arrow_len * np.sin(right_angle)

    arrowprops = dict(arrowstyle='->', color='#FF3322', linewidth=2)

    ax.annotate("", xytext=(a+left_dx, left_limit+left_dy), xy=(a, left_limit), arrowprops=arrowprops)
    ax.annotate("", xytext=(a+right_dx, right_limit+right_dy), xy=(a, right_limit), arrowprops=arrowprops)

    ax.scatter(x=a-delta, y=left_limit, color="#EE6666")

    plt.show(block=True)

def x_square(x):
    return x*x

plot_lim(x_square, a=1.5, x_min=-2, x_max=2)
