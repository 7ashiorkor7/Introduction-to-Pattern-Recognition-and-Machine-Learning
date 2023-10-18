import matplotlib.pyplot as plt
import numpy as np

# Linear solver
def my_linfit(x, y):
    a = 0
    b = 0

    sum_xi = sum(x)
    sum_yi = sum(y)
    sum_xiyi = sum(x_i * y_i for x_i, y_i in zip(x, y))
    sum_xi_squared = sum(x_i ** 2 for x_i in x)
    N = len(x)

    a = (sum_xiyi - (sum_xi * sum_yi) / N) / (sum_xi_squared - (sum_xi ** 2) / N)
    b = (sum_yi - a * sum_xi) / N

    return a, b

user_x = []
user_y = []


def onclick(event):
    if event.button == 1: 
        user_x.append(event.xdata)
        user_y.append(event.ydata)
        plt.plot(event.xdata, event.ydata, 'bo')
        plt.draw()
    elif event.button == 3:  
        if len(user_x) < 2:
            print("Please select at least two points for linear fitting.")
        else:
            a, b = my_linfit(user_x, user_y)
            plt.plot(user_x, user_y, 'bo', label='User Points')
            xp = np.arange(min(user_x), max(user_x) + 1, 0.1)
            plt.plot(xp, a * xp + b, 'r-', label=f'Linear Fit: a={a:.2f}, b={b:.2f}')
            plt.legend()
            plt.title('User Points and Linear Fit')
            plt.show()


fig, ax = plt.subplots()
ax.set_title('Click to Add Points (Right-click to fit)')


cid = fig.canvas.mpl_connect('button_press_event', onclick)


plt.show()
