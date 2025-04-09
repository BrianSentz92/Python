import pygame
import random

# Initialize Pygame
pygame.init()

# Set up display
WIDTH, HEIGHT = 960, 540
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Breakout Game")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 100, 255)
RED = (255, 50, 50)
GREEN = (50, 255, 100)
YELLOW = (255, 255, 100)
ORANGE = (255, 165, 0)

# Fonts
font = pygame.font.SysFont("Arial", 24)
large_font = pygame.font.SysFont("Arial", 36)

# Clock
clock = pygame.time.Clock()
FPS = 60

# Game variables
lives = 3
score = 0
level = 1
waiting_for_next_level = False
fireball_mode = False

# Paddle
paddle_width, paddle_height = 100, 15
paddle = pygame.Rect(WIDTH // 2 - paddle_width // 2, HEIGHT - 40, paddle_width, paddle_height)
paddle_speed = 7

# Ball
ball_radius = 10
ball_speed = [4, -4]
balls = [pygame.Rect(WIDTH // 2, HEIGHT // 2, ball_radius * 2, ball_radius * 2)]

# Bricks
brick_rows = 6
brick_cols = 12
brick_width = WIDTH // brick_cols
brick_height = 30
brick_padding = 2

def create_bricks():
    bricks = []
    for row in range(brick_rows):
        for col in range(brick_cols):
            brick = pygame.Rect(
                col * brick_width + brick_padding,
                row * brick_height + brick_padding + 60,
                brick_width - brick_padding * 2,
                brick_height - brick_padding * 2
            )
            bricks.append(brick)
    return bricks

bricks = create_bricks()

# Power-ups
powerups = []
powerup_size = 20
powerup_speed = 3
powerup_types = ["paddle_expand", "multi_ball", "fireball"]

class PowerUp:
    def __init__(self, x, y, type):
        self.rect = pygame.Rect(x, y, powerup_size, powerup_size)
        self.type = type
        self.color = YELLOW if type == "paddle_expand" else ORANGE if type == "multi_ball" else RED

    def draw(self):
        pygame.draw.rect(screen, self.color, self.rect)

    def update(self):
        self.rect.y += powerup_speed

# Main loop
running = True
while running:
    clock.tick(FPS)
    screen.fill(BLACK)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT] and paddle.left > 0:
        paddle.x -= paddle_speed
    if keys[pygame.K_RIGHT] and paddle.right < WIDTH:
        paddle.x += paddle_speed

    if waiting_for_next_level:
        text = large_font.render("Level Complete! Press N to continue", True, WHITE)
        screen.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT // 2 - 50))
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_n:
                    level += 1
                    bricks = create_bricks()
                    paddle.width = 100
                    fireball_mode = False
                    balls = [pygame.Rect(WIDTH // 2, HEIGHT // 2, ball_radius * 2, ball_radius * 2)]
                    ball_speed = [4, -4]
                    waiting_for_next_level = False
        continue

    for ball in balls[:]:
        ball.x += ball_speed[0]
        ball.y += ball_speed[1]

        if ball.left <= 0 or ball.right >= WIDTH:
            ball_speed[0] *= -1
        if ball.top <= 0:
            ball_speed[1] *= -1
        if ball.bottom >= HEIGHT:
            balls.remove(ball)
            if not balls:
                lives -= 1
                balls = [pygame.Rect(WIDTH // 2, HEIGHT // 2, ball_radius * 2, ball_radius * 2)]
                ball_speed = [4, -4]
                fireball_mode = False
                paddle.width = 100
                if lives == 0:
                    text = large_font.render("Game Over! Press R to Restart", True, RED)
                    screen.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT // 2))
                    pygame.display.flip()
                    pygame.time.wait(1000)
                    waiting = True
                    while waiting:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                running = False
                                waiting = False
                            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                                lives = 3
                                score = 0
                                level = 1
                                bricks = create_bricks()
                                paddle.width = 100
                                balls = [pygame.Rect(WIDTH // 2, HEIGHT // 2, ball_radius * 2, ball_radius * 2)]
                                ball_speed = [4, -4]
                                fireball_mode = False
                                waiting = False
                    continue

        if ball.colliderect(paddle):
            ball_speed[1] *= -1

        for brick in bricks[:]:
            if ball.colliderect(brick):
                if fireball_mode:
                    bricks.remove(brick)
                else:
                    bricks.remove(brick)
                    ball_speed[1] *= -1
                score += 10
                if random.random() < 0.2:
                    p_type = random.choice(powerup_types)
                    powerups.append(PowerUp(brick.x + brick.width // 2, brick.y, p_type))
                break

    for powerup in powerups[:]:
        powerup.update()
        powerup.draw()
        if powerup.rect.colliderect(paddle):
            if powerup.type == "paddle_expand":
                paddle.width += 30
            elif powerup.type == "multi_ball":
                new_ball = pygame.Rect(paddle.centerx, paddle.top - ball_radius * 2, ball_radius * 2, ball_radius * 2)
                balls.append(new_ball)
            elif powerup.type == "fireball":
                fireball_mode = True
            powerups.remove(powerup)
        elif powerup.rect.top > HEIGHT:
            powerups.remove(powerup)

    if not bricks:
        waiting_for_next_level = True

    pygame.draw.rect(screen, BLUE, paddle)
    for ball in balls:
        pygame.draw.ellipse(screen, RED if fireball_mode else WHITE, ball)
    for brick in bricks:
        pygame.draw.rect(screen, GREEN, brick)

    score_text = font.render(f"Score: {score}", True, WHITE)
    lives_text = font.render(f"Lives: {lives}", True, WHITE)
    level_text = font.render(f"Level: {level}", True, WHITE)
    screen.blit(score_text, (10, 10))
    screen.blit(lives_text, (WIDTH - 120, 10))
    screen.blit(level_text, (WIDTH // 2 - 40, 10))

    pygame.display.flip()

pygame.quit()
