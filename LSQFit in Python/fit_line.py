""" 1. Linear model fit with N > 2 training points 
(a) During the lecture we started to derive solutions for the parameters a and b of the linear model
y = ax + b and for N training samples {(x1, y1),(x2, y2), . . . ,(xN , yN )}.
You should first finish the derivation. Do not google, but allow yourself to do the math since errors
will be found in the next steps.
(b) Implement a Python function my linfit(x,y) that solves and returns a and b. Use your own derivations
in the function - no matter how “ugly” they are - to convince yourself about your super powers.
(c) Write a Python program that asks user to give N points with a mouse (left click: add point, right
click: stop collecting) and then plots the points and a fitted linear model. """

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
