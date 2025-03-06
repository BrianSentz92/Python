from turtle import Turtle

class Paddle(Turtle):
    def __init__(self):
        super().__init__()
        self.shape("square")
        self.color("white")  # Changed color to white
        self.shapesize(stretch_wid=5, stretch_len=1)  # Paddle size 20x100 (each unit is 20 pixels)
        self.penup()

    def move_up(self):  # Moves the paddle up
        y = self.ycor()  # Get the current y-coordinate
        if y < 250:  # Prevent moving out of the screen
            self.sety(y + 30)  # Increased the movement step to 30

    def move_down(self):  # Moves the paddle down
        y = self.ycor()  # Get the current y-coordinate
        if y > -250:  # Prevent moving out of the screen
            self.sety(y - 30)  # Increased the movement step to 30

class LeftPaddle(Paddle):
    def __init__(self):
        super().__init__()
