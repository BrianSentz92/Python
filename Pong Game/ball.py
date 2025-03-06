from turtle import Turtle

class Ball(Turtle):
    def __init__(self):
        super().__init__()
        self.shape("square")
        self.color("red")  # Changed color to red
        self.penup()
        self.speed(0)  # Make sure it doesn't have animation (immediate moves)
        self.goto(0, 0)  # Start at the center
        self.x_move = 5  # Slowed down the ball a bit for smoother gameplay
        self.y_move = 5  # Slowed down the ball a bit for smoother gameplay

    def move(self):
        """Move the ball"""
        new_x = self.xcor() + self.x_move
        new_y = self.ycor() + self.y_move
        self.goto(new_x, new_y)

    def bounce_y(self):
        """Bounce ball off top or bottom"""
        self.y_move *= -1  # Reverse direction on the y-axis

    def bounce_x(self):
        """Bounce ball off paddles"""
        self.x_move *= -1  # Reverse direction on the x-axis

    def reset_position(self):
        """Reset ball to center if it goes out of bounds"""
        self.goto(0, 0)
        self.x_move *= -1  # Reverse the direction when it resets
