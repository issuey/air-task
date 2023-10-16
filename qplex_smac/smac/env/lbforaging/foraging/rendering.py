"""
2D rendering of the level based foraging domain
"""

import math
import os
import sys

import numpy as np
import math
import six
from gym import error

if "Apple" in sys.version:
    if "DYLD_FALLBACK_LIBRARY_PATH" in os.environ:
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] += ":/usr/lib"
        # (JDS 2016/04/15): avoid bug on Anaconda 2.3.0 / Yosemite


try:
    import pyglet
except ImportError as e:
    raise ImportError(
        """
    Cannot import pyglet.
    HINT: you can install pyglet directly via 'pip install pyglet'.
    But if you really just want to install all Gym dependencies and not have to think about it,
    'pip install -e .[all]' or 'pip install gym[all]' will do it.
    """
    )

try:
    from pyglet.gl import *
except ImportError as e:
    raise ImportError(
        """
    Error occured while running `from pyglet.gl import *`
    HINT: make sure you have OpenGL install. On Ubuntu, you can run 'apt-get install python-opengl'.
    If you're running on a server, you may need a virtual frame buffer; something like this should work:
    'xvfb-run -s \"-screen 0 1400x900x24\" python <your_script.py>'
    """
    )


RAD2DEG = 57.29577951308232
# # Define some colors
# _BLACK = (0, 0, 0) # 背景颜色
# _WHITE = (255, 255, 255) # 人和线的颜色
_BLACK = (255, 255, 255) # 背景颜色
_WHITE = (0, 0, 0) # 人和线的颜色
_GREEN = (0, 255, 0)
_RED = (255, 0, 0)


def get_display(spec):
    """Convert a display specification (such as :0) into an actual Display
    object.
    Pyglet only supports multiple Displays on Linux.
    """
    if spec is None:
        return None
    elif isinstance(spec, six.string_types):
        return pyglet.canvas.Display(spec)
    else:
        raise error.Error(
            "Invalid display specification: {}. (Must be a string like :0 or None.)".format(
                spec
            )
        )


class Viewer(object):
    def __init__(self, world_size, sight):
        display = get_display(None)
        self.rows, self.cols = world_size
        # 添加观测范围
        self.sight = sight

        self.grid_size = 50
        self.icon_size = 20

        self.width = self.cols * self.grid_size + 1
        self.height = self.rows * self.grid_size + 1
        self.window = pyglet.window.Window(
            width=self.width, height=self.height, display=display
        )
        self.window.on_close = self.window_closed_by_user
        self.isopen = True

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        script_dir = os.path.dirname(__file__)

        pyglet.resource.path = [os.path.join(script_dir, "icons")]
        pyglet.resource.reindex()

        self.img_apple = pyglet.resource.image("登山.png")
        self.img_agent = pyglet.resource.image("无人机.png")

    def close(self):
        self.window.close()

    def window_closed_by_user(self):
        self.isopen = False
        exit()

    def set_bounds(self, left, right, bottom, top):
        assert right > left and top > bottom
        scalex = self.width / (right - left)
        scaley = self.height / (top - bottom)
        self.transform = Transform(
            translation=(-left * scalex, -bottom * scaley), scale=(scalex, scaley)
        )

    def render(self, env, return_rgb_array=False):
        # glClearColor(0, 0, 0, 0) # 设置窗口背景颜色
        glClearColor(255, 255, 255, 0) # 设置窗口背景颜色

        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()

        self._draw_grid()
        self._draw_food(env)
        self._draw_players(env)

        if return_rgb_array:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
            arr = arr.reshape(buffer.height, buffer.width, 4)
            arr = arr[::-1, :, 0:3]
        self.window.flip()
        return arr if return_rgb_array else self.isopen
# 绘制网格线
    def _draw_grid(self):
        # pyglet官方文档：https://pyglet.readthedocs.io/en/latest/
        batch = pyglet.graphics.Batch()
        for r in range(self.rows + 1):
            batch.add(
                2,
                gl.GL_LINES,
                None,
                (
                    "v2f",
                    (
                        0,
                        self.grid_size * r,
                        self.grid_size * self.cols,
                        self.grid_size * r,
                    ),
                ),
                ("c3B", (*_WHITE, *_WHITE)),
            )
        for c in range(self.cols + 1):
            batch.add(
                2,
                gl.GL_LINES,
                None,
                (
                    "v2f",
                    (
                        self.grid_size * c,
                        0,
                        self.grid_size * c,
                        self.grid_size * self.rows,
                    ),
                ),
                ("c3B", (*_WHITE, *_WHITE)),
            )
        # 下面是新添加的
        batch.add(
                2,
                gl.GL_LINES,
                None,
                (
                    "v2f",
                    (
                        1,
                        0,
                        1,
                        self.grid_size * self.rows,
                    ),
                ),
                ("c3B", (*_WHITE, *_WHITE)),
            )
        batch.draw()
        
    def _draw_food(self, env):
        idxes = list(zip(*env.field.nonzero()))
        apples = []
        batch = pyglet.graphics.Batch()

        # print(env.field)
        for row, col in idxes:
            apples.append(
                pyglet.sprite.Sprite(
                    self.img_apple,
                    self.grid_size * col,
                    self.height - self.grid_size * (row + 1),
                    batch=batch,
                )
            )
        for a in apples:
            a.update(scale=self.grid_size / a.width)
        batch.draw()

        for row, col in idxes:
            self._draw_badge(row, col, env.field[row, col])

    def _draw_players(self, env):
        players = []
        batch = pyglet.graphics.Batch()
        # colors = [(186, 208, 239, 100), (221, 237, 221, 100),(235, 217, 229,100),(255, 245, 214,100)]

        for i,player in enumerate(env.players):
            row, col = player.position
            players.append(
                pyglet.sprite.Sprite(
                    self.img_agent,
                    self.grid_size * col,
                    self.height - self.grid_size * (row + 1),
                    batch=batch,
                )
            )
            # 添加绘制观测范围的代码
             # 在 Batch 中添加一个带有透明度的矩形
            batch.add(
                4,
                GL_QUADS,
                None,
                
                # ("v2f", (100, 100, 200, 100, 200, 200, 100, 200)),
                ("v2f", (
                    self.grid_size * (col-self.sight), self.height - self.grid_size * (row + self.sight+1),# 左下
                    self.grid_size * (col+self.sight+1), self.height - self.grid_size * (row + self.sight+1),# 右下
                    self.grid_size * (col+self.sight+1), self.height - self.grid_size * (row - self.sight),# 右上
                    self.grid_size * (col-self.sight), self.height - self.grid_size * (row - self.sight)# 左上
                )),
                ("c4B", (232, 133, 101,40) * 4),# 最后一个是矩形的不透明度(0 = invisible, 255 means visible)
            )
            # rec1 = pyglet.shapes.Rectangle(self.grid_size * (col-self.sight), self.height - self.grid_size * (row + self.sight)
            #                                , self.grid_size * (self.sight*2+1), self.grid_size * (self.sight*2+1), color = (61, 210, 247), batch = batch)

            
            # rec1.opacity = 250
        for p in players:
            p.update(scale=self.grid_size / p.width)
            
        batch.draw()
        for p in env.players:
            self._draw_badge(*p.position, p.level)

    def _draw_badge(self, row, col, level):
        resolution = 6
        radius = self.grid_size / 5

        badge_x = col * self.grid_size + (3 / 4) * self.grid_size
        badge_y = self.height - self.grid_size * (row + 1) + (1 / 4) * self.grid_size

        # make a circle
        verts = []
        for i in range(resolution):
            angle = 2 * math.pi * i / resolution
            x = radius * math.cos(angle) + badge_x
            y = radius * math.sin(angle) + badge_y
            verts += [x, y]
        circle = pyglet.graphics.vertex_list(resolution, ("v2f", verts))
        glColor3ub(*_BLACK)
        circle.draw(GL_POLYGON)
        glColor3ub(*_WHITE)
        circle.draw(GL_LINE_LOOP)
        label = pyglet.text.Label(
            str(level),
            font_name="Times New Roman",
            font_size=12,
            x=badge_x,
            y=badge_y + 2,
            anchor_x="center",
            anchor_y="center",
            color=(0,0,0,255)
        )
        label.draw()
