from math import sqrt

class Entity:

    def __init__(self, x, y, size, vx=0, vy=0):
        self.x = x
        self.y = y
        self.size = size
        self.vx = vx
        self.vy = vy

        self.age = 0

    # Square collision bounding boxes
    def collides(self, other):
        if self.x > other.x + other.size or other.x > self.x + self.size:
            return False

        if self.y + self.size < other.y or other.y + other.size < self.y:
            return False

        return True

    def update(self):
        self.x += self.vx
        self.y += self.vy

        self.age += 1

    @property
    def position(self):
        return self.x, self.y

    @property
    def should_delete(self):
        return self.age >= 2 * int(sqrt(2 * (self.size ** 2))) + 1
