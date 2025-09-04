import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.widgets import Slider
import numpy as np

plt.ion()

colors = [
    '#1f77b4',
    '#ff7f0e',
    '#2ca02c',
    '#9467bd',
    '#ffc107',
    '#17becf',
    '#e377c2',
    '#8c564b',
    '#7f7f7f',
]

_curve_color_index = len(colors) - 1
def get_curve_color():
    global _curve_color_index

    _curve_color_index = (_curve_color_index + 1) % len(colors)
    return colors[_curve_color_index]

def plot_limit(f, a, delta, epsilon, x_min=None, x_max=None, y_min=None, y_max=None, ax=None, color=None):
    if color is None:
        color = get_curve_color()

    if ax is None:
        fig, ax = plt.subplots()

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

    f_y_min = np.min(y)
    f_y_max = np.max(y)
    f_y_amp = np.abs(f_y_min - f_y_max)

    if y_min is None:
        y_min = (f_y_min + f_y_max)/2 - f_y_amp
    if y_max is None:
        y_max = (f_y_min + f_y_max)/2 + f_y_amp

    ax.grid(visible=True, color='#DDDDDD', linestyle='--')

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

    ax.plot(x, y, color=color, label='y = f(x)')

    # arrows on the x axis
    ax.annotate("", xytext=(a-delta-arrow_len, y_min - x_arrow_margin), xy=(a-delta, y_min - x_arrow_margin), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xytext=(a+delta+arrow_len, y_min - x_arrow_margin), xy=(a+delta, y_min - x_arrow_margin), arrowprops=dict(arrowstyle="->"))

    _l_delta = 0.00001 # to compute limits, not the one specified by the user
    L, left_limit, right_limit = limit(f, a)

    axis_ratio = x_amp/y_amp

    # compute left and right derivatives
    _, dy_dx_left, _ = derivative(f, a-delta)
    _, _, dy_dx_right = derivative(f, a+delta)
    dy_dx, _, _ = derivative(f, a)

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

    x_pm_delta = x[(x >= a-delta) & (x <= a+delta)]
    f_x_pm_delta = f(x_pm_delta)

    vertices_under = [[x_pm_delta[0], y_min-y_margin]]
    vertices_under.extend(list(a) for a  in zip(x_pm_delta, f(x_pm_delta)))
    vertices_under.append([x_pm_delta[-1], y_min-y_margin])
    vertices_under.append([x_pm_delta[0], y_min-y_margin])

    for x1, x2 in zip(x_pm_delta, x_pm_delta[1:]):
        y1, y2 = f(np.array([x1]))[0], f(np.array([x2]))[0]
        x1, y1, x2, y2 = (x1, y1, x2, y2) if y1 < y2 else (x2, y2, x1, y1)

        vertices = [[x_min, y1], [x1, y1], [x2, y2], [x_min, y2], [x_min, y1]]

        ax.add_patch(Polygon(vertices, closed=True, facecolor='lightblue', edgecolor='lightblue', alpha=1.0))

    polygon_under = Polygon(vertices_under, closed=True, facecolor='lightblue', edgecolor='lightblue', alpha=1.0)
    ax.add_patch(polygon_under)

    # L +- epsilon
    epsilon_left_vertices = [[x_min, left_limit-epsilon], [x_max, left_limit-epsilon],
                             [x_max, left_limit+epsilon], [x_min, left_limit+epsilon], 
                             [x_min, left_limit-epsilon]]
    polygon_epsilon_left = Polygon(epsilon_left_vertices, closed=True, facecolor='#FF3355', edgecolor='#FF3355', alpha=0.1)
    ax.add_patch(polygon_epsilon_left)

    epsilon_right_vertices = [[x_min, right_limit-epsilon], [x_max, right_limit-epsilon],
                             [x_max, right_limit+epsilon], [x_min, right_limit+epsilon], 
                             [x_min, right_limit-epsilon]]
    polygon_epsilon_right = Polygon(epsilon_right_vertices, closed=True, facecolor='#FF3355', edgecolor='#FF3355', alpha=0.1)
    ax.add_patch(polygon_epsilon_right)

    ax.scatter(x=a, y=left_limit, color="#EE6666")
    ax.scatter(x=a, y=right_limit, color="#EE6666")

_sliders = []
def plot_limit_slider(f, a, delta, epsilon, x_min=None, x_max=None, y_min=None, y_max=None, ax=None, color=None,
               delta_min=1e-3, delta_max=1e-1, epsilon_min=1e-3, epsilon_max=1e-1):

    delta_epsilon = dict(delta=delta, epsilon=epsilon)

    global _sliders

    if color is None:
        color = get_curve_color()

    ax_delta = plt.axes([0.1, 0.04, 0.35, 0.03])
    delta_slider = Slider(ax_delta, 'Delta', delta_min, delta_max, valinit=delta_epsilon['delta'])

    ax_epsilon = plt.axes([0.1, 0.01, 0.35, 0.03])
    epsilon_slider = Slider(ax_epsilon, 'Epsilon', epsilon_min, epsilon_max, valinit=delta_epsilon['epsilon'])

    plot_limit(f, a=a, delta=delta_epsilon['delta'], epsilon=delta_epsilon['epsilon'], x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, ax=ax, color=color)

    def update_delta(new_delta):
        ax.clear()
        delta_epsilon['delta'] = new_delta
        plot_limit(f, a=a, delta=delta_epsilon['delta'], epsilon=delta_epsilon['epsilon'], x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, ax=ax, color=color)

    def update_epsilon(new_epsilon):
        ax.clear()
        delta_epsilon['epsilon'] = new_epsilon
        plot_limit(f, a=a, delta=delta_epsilon['delta'], epsilon=delta_epsilon['epsilon'], x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, ax=ax, color=color)

    delta_slider.on_changed(update_delta)
    epsilon_slider.on_changed(update_epsilon)

    # avoid sliders being garbage collected to keep them working after function execution
    _sliders.append(delta_slider)
    _sliders.append(epsilon_slider)


def limit(f, x, max_iters=20, presicion=1e-6):
    delta = 0.01

    prev_left_value = f(np.array([x - delta]))[0]
    left_value = prev_left_value
    for _ in range(max_iters):
        delta *= 0.1
        left_value = f(np.array([x - delta]))[0]

        if np.abs(left_value - prev_left_value) < presicion:
            break

        prev_left_value = left_value

    delta = 0.01
    prev_right_value = f(np.array([x + delta]))[0]
    right_value = prev_right_value
    for _ in range(max_iters):
        delta *= 0.1
        right_value = f(np.array([x + delta]))[0]

        if np.abs(right_value - prev_right_value) < presicion:
            break

        prev_right_value = right_value

    result = left_value if np.abs(left_value - right_value) < presicion*10 else None

    return result, left_value, right_value

def derivative(f, x, max_iters=20, presicion=1e-8):
    return limit(lambda t: (f(x+t) - f(x-t))/(2*t), 0, max_iters, presicion)

if __name__ == '__main__':
    delta = 0.1
    epsilon = 0.01
    a = 0

    f1 = lambda x: -(x-1)**2 + 4
    f2 = lambda x: x*x
    f3 = lambda x: x**3-2*x**2-x+2
    f4 = np.sign

    f = f4

    fig, axs = plt.subplots(2, 2)
    plot_limit(f1, a=a, delta=delta, epsilon=epsilon, ax=axs[0, 0])
    plot_limit(f2, a=a, delta=delta, epsilon=epsilon, ax=axs[0, 1])
    plot_limit(f3, a=a, delta=delta, epsilon=epsilon, ax=axs[1, 0])
    plot_limit(f4, a=a, delta=delta, epsilon=epsilon, ax=axs[1, 1])

    #fig, ax = plt.subplots()
    #plot_limit_slider(f1, a=a, delta=delta, epsilon=epsilon, ax=ax)

    plt.show(block=True)
