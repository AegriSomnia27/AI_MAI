import numpy as np
import pygame as pg
import math
import numba

# settings
res = W, H = 800, 450
offset = np.array([1.3 * W, H]) // 2
max_iter = 30
zoom = 2.2 / H

# texture
texture = pg.image.load('texture_2.jpg')
texture_size = min(texture.get_size()) - 1
texture_array = pg.surfarray.array3d(texture)


class Fractal:
    def __init__(self, app):
        self.app = app
        self.screen_array = np.full((W, H, 3), [0, 0, 0], dtype=np.uint8)
        self.x = np.linspace(0, W, num=W, dtype=np.float32)
        self.y = np.linspace(0, H, num=H, dtype=np.float32)

    @staticmethod
    @numba.njit(fastmath=True)
    def render(screen_array):
        # x = (self.x - offset[0]) * zoom
        # y = (self.y - offset[1]) * zoom
        # c = x + 1j * y[:, None]
        #
        # num_iter = np.full(c.shape, max_iter)
        # z = np.empty(c.shape, np.complex64)
        # for i in range(max_iter):
        #     mask = (num_iter == max_iter)
        #     z[mask] = z[mask]**2 + c[mask]
        #     #num_iter[mask & (np.abs(z) > 2.0)] = i + 1
        #     num_iter[mask & (z.real**2 + z.imag**2 > 4.0)] = i + 1
        #
        # col = (num_iter.T * texture_size / max_iter).astype(np.uint8)
        # self.screen_array = texture_array[col, col]

        for x in range(W):
            for y in range(H):
                c = (x - offset[0]) * zoom + 1j * (y - offset[1]) * zoom
                z = 0
                num_iter = 0
                for i in range(max_iter):
                    z = z**2 + c
                    if z.real**2 + z.imag**2 > 4:
                        break
                    num_iter += 1
                col = int(texture_size * num_iter / max_iter)
                screen_array[x, y] = texture_array[col, col]
        return screen_array

    def update(self):
        self.screen_array = self.render(self.screen_array)

    def draw(self):
        pg.surfarray.blit_array(self.app.screen, self.screen_array)

    def run(self):
        self.update()
        self.draw()


class App:
    def __init__(self):
        self.screen = pg.display.set_mode(res, pg.SCALED)
        self.clock = pg.time.Clock()
        self.fractal = Fractal(self)

    def run(self):
        while True:
            self.screen.fill('black')
            self.fractal.run()
            pg.display.flip()

            [exit() for i in pg.event.get() if i.type == pg.QUIT]
            self.clock.tick()
            pg.display.set_caption(f'FPS: {self.clock.get_fps()}')


if __name__ == '__main__':
    app = App()
    app.run()
