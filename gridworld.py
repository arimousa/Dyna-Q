import numpy as np

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class GridworldEnv:
    def __init__(self, shape=[6, 9]):
        if not isinstance(shape, (list, tuple)) or not len(shape) == 2:
            raise ValueError('shape argument must be a list/tuple of length 2')

        self.shape = shape

        self.nS = np.prod(shape)
        self.nA = 4

        self.walls = np.zeros((shape))
        self.walls[0:2, 7] = 1
        self.walls[1:3, 2] = 1
        self.walls[4, 5] = 1

    def next_state(self, s, a):
        x, y = np.unravel_index([s], self.shape)

        if a == UP:
            nx = x - 1
            ny = y
        elif a == RIGHT:
            nx = x
            ny = y + 1
        elif a == DOWN:
            nx = x + 1
            ny = y
        elif a == LEFT:
            nx = x
            ny = y - 1

        if nx >= self.shape[0] or nx < 0 or ny >= self.shape[1] or ny < 0:
            nx = x
            ny = y

        if self.walls[nx, ny]:
            nx = x
            ny = y

        ns = np.ravel_multi_index([nx, ny], self.shape)[0]
        return ns

    def step(self, a):
        ns = self.next_state(self.s, a)
        d = ns == 8
        r = 1.0 if d else 0

        self.s = ns
        return ns, r, d

    def reset(self):
        self.s = np.ravel_multi_index([2, 0], self.shape)
        return self.s
