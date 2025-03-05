from turtle import Turtle

ALIGNMENT = "center"
FONT = ("Courier", 24, "normal")

class Scoreboard(Turtle):
    def __init__(self):
        super().__init__()
        self.score = 0
        self.color("white")
        self.penup()
        self.hideturtle()
        self.goto(0, 260)
        self.update_scoreboard()

    def update_scoreboard(self):
        """Updates the score display."""
        self.clear()
        self.write(f"Score: {self.score}", align=ALIGNMENT, font=FONT)

    def increase_score(self):
        """Increases score and updates display."""
        self.score += 1
        self.update_scoreboard()

    def game_over(self):
        """Displays the Game Over message."""
        self.goto(0, 0)
        self.write("GAME OVER", align=ALIGNMENT, font=FONT)
