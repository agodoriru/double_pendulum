import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from matplotlib.animation import FuncAnimation
from numpy import sin, cos

G = 9.80665  # gravitational acceleration
L = 1.00


class SinglePendulum:
    def __init__(self, theta_0, w_0, dt):
        self.theta_0 = theta_0  # initial angle [deg]
        self.w_0 = w_0  # initial angular velocity [deg/s]
        self.dt = dt  # interval
        self.t = np.arange(0.00, 10, dt)  # 0 - t dt interval

        # initial1 state
        self.initial_state = np.radians([theta_0, w_0])  # convert [deg] to [rad]

        # graph setting
        self.fig = plt.figure()
        ax = self.fig.add_subplot(111,
                                  autoscale_on=False,
                                  xlim=(-1.5 * L, 1.5 * L),
                                  ylim=(-1.5 * L, 1.5 * L))
        ax.set_aspect('equal')
        ax.grid()

        self.line, = ax.plot([], [], 'o-')

    def func(self, state, t):
        dwdt = np.zeros_like(state)
        dwdt[0] = state[1]
        dwdt[1] = -(G / L) * sin(state[0])

        return dwdt

    def solver(self):
        sol = odeint(self.func, self.initial_state, self.t)
        return sol

    def coordinate_conversion(self):
        theta = self.solver()[:, 0]
        # print(theta)

        x = L * sin(theta)  # cos(theta - 1/2 * pi)
        y = -L * cos(theta)  # sin(theta - 1/2 * pi)

        return [x, y]

    def animate(self, i):
        xy = self.coordinate_conversion()

        x = [0, xy[0][i]]
        y = [0, xy[1][i]]

        self.line.set_data(x, y)
        return self.line,

    def plot(self):
        ani = FuncAnimation(self.fig,
                            self.animate,
                            frames=np.arange(0, len(self.t)),
                            interval=25,
                            blit=True)
        plt.show()


if __name__ == '__main__':
    hoge = SinglePendulum(30, 0, 0.01)
    hoge.plot()
