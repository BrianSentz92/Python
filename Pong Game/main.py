import turtle
from paddle import Paddle, LeftPaddle
from ball import Ball

# Set up the screen
screen = turtle.Screen()
screen.bgcolor("black")  # Black background
screen.setup(width=800, height=600)  # Set window size
screen.title("Brian's Pong Game")  # Window title

# Create paddles and set their starting positions
right_paddle = Paddle()
right_paddle.goto(350, 0)  # Right paddle starts on the right side

left_paddle = LeftPaddle()
left_paddle.goto(-350, 0)  # Left paddle starts on the left side

# Create ball
ball = Ball()

# Define movement functions for the paddles
def move_right_up():
    right_paddle.move_up()

def move_right_down():
    right_paddle.move_down()

def move_left_up():
    left_paddle.move_up()

def move_left_down():
    left_paddle.move_down()

# Listen for keyboard inputs (Arrow keys for right paddle, W/S for left paddle)
screen.listen()
screen.onkey(move_right_up, "Up")
screen.onkey(move_right_down, "Down")
screen.onkey(move_left_up, "w")
screen.onkey(move_left_down, "s")

# Game loop
while True:
    screen.update()  # Update the screen for every frame

    # Move the ball
    ball.move()

    # Ball collision with top and bottom
    if ball.ycor() > 290:  # If ball hits the top wall
        ball.bounce_y()
    elif ball.ycor() < -290:  # If ball hits the bottom wall
        ball.bounce_y()

    # Ball collision with paddles (right and left)
    # Right paddle (position x > 340 and x < 350)
    if ball.xcor() > 340 and ball.xcor() < 350:
        if right_paddle.ycor() + 50 > ball.ycor() > right_paddle.ycor() - 50:  # Check if ball is within paddle height
            ball.bounce_x()

    # Left paddle (position x < -340 and x > -350)
    if ball.xcor() < -340 and ball.xcor() > -350:
        if left_paddle.ycor() + 50 > ball.ycor() > left_paddle.ycor() - 50:  # Check if ball is within paddle height
            ball.bounce_x()

    # Ball out of bounds (right or left)
    if ball.xcor() > 390:  # Right side out of bounds
        ball.reset_position()
    elif ball.xcor() < -390:  # Left side out of bounds
        ball.reset_position()
