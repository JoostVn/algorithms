import pygame
import time
import numpy as np
from algorithms.geometry.shapes import Line, Polygon, Circle
from math import pi, sin, cos
import pygame.gfxdraw


class Sprite:

    COL = [
        None,
        None,
        None]

    def __init__(self, window_size, size):
        self.size = size

        # Shape highlighting
        self.hl = [0, 0]

        self.shape = None
        self.pos_direction = None
        self.ang_direction = None

        self.create_shape(window_size)
        self.randomize_movement()

    def __repr__(self):
        return str(self.shape)

    def create_shape(self, window_size):
        """
        Implement in sub class.
        """
        raise NotImplementedError

    def draw_shape(self, screen, color):
        """
        Implement in sub class.
        """
        raise NotImplementedError

    def draw(self, screen, highlight_type=None):
        """
        Determines the color for drawing the shape to the screen, and calls
        self.drawshape. The highlight_type parameter makes the shape darker.
        """
        if highlight_type in [0,1]:
            self.hl[highlight_type] = 1
        col = self.COL[int(sum(self.hl))]
        self.draw_shape(screen, col)

    def randomize_movement(self, speed=2):
        neg_pos = np.array([-1,1])
        self.pos_direction = np.random.uniform(*(0.5 * speed * neg_pos),2)
        self.ang_direction = np.random.uniform(*(0.01 * speed * neg_pos))

    def update_pos(self):

        # Reset highlights
        self.hl = [0, 0]

        # Randomize movement
        if np.random.uniform(0,1) < 0.01:
            self.randomize_movement()

        # Tranform shape
        self.shape.translate(self.pos_direction)
        self.shape.rotate(self.ang_direction, self.shape.center())



class SpriteLine(Sprite):

    COL = [
        (250,210,210),
        (200,110,110),
        (50,0,0)]

    def create_shape(self, window_size):
        p1 = np.random.uniform((0,0), window_size, 2)
        p2 = p1 + np.random.choice((-1,1)) * np.random.uniform(self.size/4, self.size, 2)
        points = np.vstack((p1,p2))
        self.shape = Line(points)

    def draw_shape(self, screen, color):
        p1, p2 = self.shape.points.astype(int)
        pygame.gfxdraw.aapolygon(screen, (p1, p2, p1), color)



class SpritePolygon(Sprite):

    COL = [
        (210,210,250),
        (90,90,190),
        (30,30,80)]

    def create_shape(self, window_size):
        mid = np.random.uniform((0,0), window_size, 2)
        nr_points = np.random.randint(3, 8)
        angles = np.linspace(0, 2*pi, nr_points, False)
        points = np.vstack([np.add(mid, [self.size*cos(a), self.size*sin(a)]) for a in angles])
        points = points + np.random.uniform(0, self.size, (nr_points, 2))
        self.shape = Polygon(points)

    def draw_shape(self, screen, color):
        pygame.gfxdraw.aapolygon(screen, self.shape.points, color)

    def draw_interior(self, screen):
        pygame.gfxdraw.filled_polygon(screen, self.shape.points, (245,245,250))


class PygameScreen:

    BLACK = (60,60,60)
    RED = (250, 160, 160)
    GREEN = (120,250,120)
    BLUE = (180,180,250)

    def __init__(self, window_size, n_lines, n_pols, tick_len=1/30):
        self.bg = (255,255,255)
        self.tick_len = tick_len
        self.window_size = window_size
        self.screen = pygame.display.set_mode(window_size)
        self.lines = [SpriteLine(self.window_size, 240) for i in range(n_lines)]
        self.pols = [SpritePolygon(self.window_size, 60) for i in range(n_pols)]

    def draw_marker(self, pos, color, markertype):
        if markertype == 'inner':
            pygame.gfxdraw.filled_circle(self.screen, *pos.astype(int), 3, color)
            pygame.gfxdraw.aacircle(self.screen, *pos.astype(int), 3, color)
        elif markertype == 'outer':
            pygame.gfxdraw.filled_circle(self.screen, *pos.astype(int), 4, color)
            pygame.gfxdraw.aacircle(self.screen, *pos.astype(int), 4, color)


    def draw_intersections(self):
        """
        Draw intersections. All intersections are tested both ways, from
        shape1 -> shape2 and vice versa. First intersection is drawn black
        and bigger, and the second intersection is draw over the first
        as a smaller colored circle. So, only colored intersections with
        a black outline are valid.
        """
        ############################## INTERSECTION BOOL (COLORING)

        # Line line bool
        for i1, line1 in enumerate(self.lines):
            for i2, line2 in enumerate(self.lines):
                if i1 < i2:
                    if line1.shape.intersect_line_bool(line2.shape):
                        line1.draw(self.screen, 0)
                    if line2.shape.intersect_line_bool(line1.shape):
                        line2.draw(self.screen, 0)

        # Line pol bool
        for line in self.lines:
            for pol in self.pols:
                if line.shape.intersect_pol_bool(pol.shape):
                    line.draw(self.screen, 1)
                if pol.shape.intersect_line_bool(line.shape):
                    pol.draw(self.screen, 1)

        # Pol pol bool
        for i1, pol1 in enumerate(self.pols):
            for i2, pol2 in enumerate(self.pols):
                if i1 < i2:
                    if pol1.shape.intersect_pol_bool(pol2.shape):
                        pol1.draw(self.screen, 0)
                    if pol2.shape.intersect_pol_bool(pol1.shape):
                        pol2.draw(self.screen, 0)

        ############################## INTERSECTION POINTS (CIRCLE MARKERS)

        # Line line intersections (red)
        for i1, line1 in enumerate(self.lines):
            for i2, line2 in enumerate(self.lines):
                if i1 < i2:
                    inter1 = line1.shape.intersect_line(line2.shape)
                    if len(inter1) > 0:
                        self.draw_marker(inter1, self.BLACK, 'outer')
                    inter2 = line2.shape.intersect_line(line1.shape)
                    if len(inter2) > 0:
                        self.draw_marker(inter2, self.RED, 'inner')

        # Line pol intersections (green)
        for line in self.lines:
            for pol in self.pols:
                pol_inters = pol.shape.intersect_line(line.shape)
                if len(pol_inters) > 0:
                    for inter in pol_inters:
                        self.draw_marker(inter, self.BLACK, 'outer')
                line_inters = line.shape.intersect_pol(pol.shape)
                if len(line_inters) > 0:
                    for inter in line_inters:
                        self.draw_marker(inter, self.GREEN, 'inner')

        # pol pol intersections (blue)
        for i1, pol1 in enumerate(self.pols):
            for i2, pol2 in enumerate(self.pols):
                if i1 < i2:
                    inters1 = pol1.shape.intersect_pol(pol2.shape)
                    if len(inters1) > 0:
                        for inter in inters1:
                            self.draw_marker(inter, self.BLACK, 'outer')
                    inters2 = pol2.shape.intersect_pol(pol1.shape)
                    if len(inters2) > 0:
                        for inter in inters2:
                            self.draw_marker(inter, self.BLUE, 'inner')

    def show(self):
        """
        Show Pygame window and plots.
        """
        tick_start = time.time()
        while True:

            # Reset sprites with outbound points
            for sprite in self.lines + self.pols:
                x, y = sprite.shape.points.T
                n = len(sprite.shape.points)
                out_bounds_x = ((x < 0) | (x > self.window_size[0])).sum()
                out_bounds_y = ((y < 0) | (y > self.window_size[1])).sum()
                if out_bounds_x == n or out_bounds_y == n:
                    sprite.create_shape(self.window_size)

            # Update and draw sprites on the screen
            self.screen.fill(self.bg)
            for pol in self.pols:
                pol.update_pos()
                pol.draw_interior(self.screen)
                pol.draw(self.screen)
            for line in self.lines:
                line.update_pos()
                line.draw(self.screen)

            self.draw_intersections()
            pygame.display.flip()

            # Pygame update stuff
            t = (time.time() - tick_start)
            time.sleep(max(0, self.tick_len - t))
            tick_start = time.time()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return



window_size = (800,600)
ps = PygameScreen(window_size, n_lines=20, n_pols=20)
ps.show()


#c = Circle((0,0), 10, 10)