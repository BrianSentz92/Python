from turtle import Turtle

MOVE_DISTANCE = 20
START_POSITIONS = [(0, 0), (-20, 0), (-40, 0)]  # Initial snake body positions

class Snake:
    def __init__(self):
        self.segments = []
        self.create_snake()
        self.head = self.segments[0]
        self.direction = "Right"

    def create_snake(self):
        """Creates the initial 3-segment snake."""
        for position in START_POSITIONS:
            self.add_segment(position)

    def add_segment(self, position):
        """Adds a segment to the snake's body."""
        segment = Turtle("square")
        segment.color("white")
        segment.penup()
        segment.goto(position)
        self.segments.append(segment)

    def grow(self):
        """Grows the snake when it eats food."""
        self.add_segment(self.segments[-1].position())

    def move(self):
        """Moves the snake forward."""
        for i in range(len(self.segments) - 1, 0, -1):  # Move each segment forward
            new_x = self.segments[i - 1].xcor()
            new_y = self.segments[i - 1].ycor()
            self.segments[i].goto(new_x, new_y)

        if self.direction == "Up":
            self.head.sety(self.head.ycor() + MOVE_DISTANCE)
        elif self.direction == "Down":
            self.head.sety(self.head.ycor() - MOVE_DISTANCE)
        elif self.direction == "Left":
            self.head.setx(self.head.xcor() - MOVE_DISTANCE)
        elif self.direction == "Right":
            self.head.setx(self.head.xcor() + MOVE_DISTANCE)

    # Methods to change direction, preventing immediate 180° turns
    def up(self):
        if self.direction != "Down":
            self.direction = "Up"

    def down(self):
        if self.direction != "Up":
            self.direction = "Down"

    def left(self):
        if self.direction != "Right":
            self.direction = "Left"

    def right(self):
        if self.direction != "Left":
            self.direction = "Right"
