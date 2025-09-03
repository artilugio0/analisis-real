import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

def plot_limit(f, a, delta, epsilon, x_min=None, x_max=None, y_min=None, y_max=None):
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

    fig, ax = plt.subplots()

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

    ax.plot(x, y, color='blue', label='y = f(x)')

    # arrows on the x axis
    ax.annotate("", xytext=(a-delta-arrow_len, y_min - x_arrow_margin), xy=(a-delta, y_min - x_arrow_margin), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xytext=(a+delta+arrow_len, y_min - x_arrow_margin), xy=(a+delta, y_min - x_arrow_margin), arrowprops=dict(arrowstyle="->"))

    _l_delta = 0.00001 # to compute limits, not the one specified by the user
    L, left_limit, right_limit = limit(f, a)
    print(L, left_limit, right_limit)

    axis_ratio = x_amp/y_amp

    # compute left and right derivatives
    _, dy_dx_left, _ = derivative(f, a-delta)
    _, _, dy_dx_right = derivative(f, a+delta)
    dy_dx, _, _ = derivative(f, a)
    print(dy_dx, dy_dx_left, dy_dx_right)
    #dy_dx_left = (f(a-delta) - f(a-delta+-_l_delta))/(_l_delta)
    #dy_dx_right = (f(a+delta) - f(a+delta+_l_delta))/(-_l_delta)

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

    plt.show(block=True)

def limit(f, x, max_iters=20, presicion=1e-6):
    delta = 0.01

    prev_left_value = f(np.array([x - delta]))[0]
    left_value = prev_left_value
    for _ in range(max_iters):
        delta *= 0.1
        left_value = f(np.array([x - delta]))[0]
        print(f'left: deta: {delta}, prev: {prev_left_value}, curr: {left_value}')

        if np.abs(left_value - prev_left_value) < presicion:
            break

        prev_left_value = left_value

    delta = 0.01
    prev_right_value = f(np.array([x + delta]))[0]
    right_value = prev_right_value
    for _ in range(max_iters):
        delta *= 0.1
        right_value = f(np.array([x + delta]))[0]
        print(f'right: deta: {delta}, prev: {prev_right_value}, curr: {right_value}')

        if np.abs(right_value - prev_right_value) < presicion:
            break

        prev_right_value = right_value

    result = left_value if np.abs(left_value - right_value) < presicion*10 else None

    return result, left_value, right_value

def derivative(f, x, max_iters=20, presicion=1e-12):
    return limit(lambda t: (f(x+t) - f(x-t))/(2*t), 0, max_iters, presicion)

if __name__ == '__main__':

    plot_limit(lambda x: -(x-1)**2 + 4, a=1, delta=0.1, epsilon=0.01)
    plot_limit(lambda x: x*x, a=1, delta=0.1, epsilon=0.01)
    plot_limit(np.sign, a=0, delta=0.1, epsilon=0.01)
    plot_limit(lambda x: x**3-2*x**2-x+2, a=0, delta=0.05, epsilon=0.01)
