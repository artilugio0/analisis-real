import matplotlib.pyplot as plt
import numpy as np

def plot_lim(f, a, delta, x_min=None, x_max=None, y_min=None, y_max=None):
    if x_min is None:
        if x_max is not None:
            x_min = a - np.abs(a - x_max)
        else:
            x_min = a - 5*delta

    if x_max is None:
        if x_min is not None:
            x_max = a + np.abs(a - x_min)
        else:
            x_max = a + 5*delta

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
    arrow_len = x_amp*0.06

    ax.set_ylim(y_min - y_margin, y_max + y_margin)

    ax.plot(x, y, color='blue', label='y = f(x)')

    # arrows on the x axis
    ax.annotate("", xytext=(a-delta-arrow_len, y_min - x_arrow_margin), xy=(a-delta, y_min - x_arrow_margin), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xytext=(a+delta+arrow_len, y_min - x_arrow_margin), xy=(a+delta, y_min - x_arrow_margin), arrowprops=dict(arrowstyle="->"))

    _l_delta = 0.00001 # to compute limits, not the one specified by the user
    left_limit = f(np.array([a-_l_delta]))[0]
    right_limit = f(np.array([a+_l_delta]))[0]

    axis_ratio = x_amp/y_amp

    # compute left and right derivatives
    dy_dx_left = (f(a-delta) - f(a-delta+-_l_delta))/(_l_delta)
    dy_dx_right = (f(a+delta) - f(a+delta+_l_delta))/(-_l_delta)

    left_angle = np.atan(dy_dx_left) + np.pi

    # factor to keep arrow of length arrow_len when the axis ratio is not 1
    left_factor = arrow_len / np.sqrt(np.cos(left_angle)**2 + np.sin(left_angle)**2 * axis_ratio**2)
    left_dx = left_factor * np.cos(left_angle)
    left_dy = left_factor * np.sin(left_angle)

    right_angle = np.atan(dy_dx_right)

    # factor to keep arrow of length arrow_len when the axis ratio is not 1
    right_factor = arrow_len / np.sqrt(np.cos(right_angle)**2 + np.sin(right_angle)**2 * axis_ratio**2)
    right_dx = right_factor * np.cos(right_angle)
    right_dy = right_factor * np.sin(right_angle)

    arrowprops = dict(arrowstyle='->', color='#FF3322', linewidth=2)

    # arrows on the curve
    ax.annotate("", xytext=(a-delta+left_dx, f(a-delta)+left_dy), xy=(a-delta, f(a-delta)), arrowprops=arrowprops)
    ax.annotate("", xytext=(a+delta+right_dx, f(a+delta)+right_dy), xy=(a+delta, f(a+delta)), arrowprops=arrowprops)

    ax.scatter(x=a-_l_delta, y=left_limit, color="#EE6666")
    ax.scatter(x=a-_l_delta, y=right_limit, color="#EE6666")

    plt.show(block=True)


def x_square(x):
    return x*x


if __name__ == '__main__':
    plot_lim(x_square, a=2, delta=0.1, x_min=0, x_max=4)
