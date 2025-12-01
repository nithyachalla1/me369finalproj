import pygame
import sys
import random

WIDTH, HEIGHT = 480, 720
FPS = 60

NUM_LANES = 3
LANE_WIDTH = WIDTH // NUM_LANES

PLAYER_WIDTH = 50
PLAYER_HEIGHT = 80
PLAYER_Y = HEIGHT - PLAYER_HEIGHT - 30

OBSTACLE_WIDTH = 50
OBSTACLE_HEIGHT = 80
OBSTACLE_SPEED = 7
OBSTACLE_SPAWN_TIME = 900

BG_COLOR = (10, 10, 25)
LANE_COLOR = (60, 60, 90)
PLAYER_COLOR = (50, 220, 70)
OBSTACLE_COLOR = (220, 50, 50)
TEXT_COLOR = (255, 255, 255)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Three-Lane Runner")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 36)
big_font = pygame.font.SysFont(None, 72)

LANE_X = []
for i in range(NUM_LANES):
    center_x = i * LANE_WIDTH + LANE_WIDTH // 2
    LANE_X.append(center_x)


class Player:
    def __init__(self):
        self.lane_idx = 1
        self.rect = pygame.Rect(0, 0, PLAYER_WIDTH, PLAYER_HEIGHT)
        self.update_position()

    def update_position(self):
        self.rect.centerx = LANE_X[self.lane_idx]
        self.rect.y = PLAYER_Y

    def move_left(self):
        if self.lane_idx > 0:
            self.lane_idx -= 1
            self.update_position()

    def move_right(self):
        if self.lane_idx < NUM_LANES - 1:
            self.lane_idx += 1
            self.update_position()

    def draw(self, surface):
        pygame.draw.rect(surface, PLAYER_COLOR, self.rect)


class Obstacle:
    def __init__(self):
        lane_idx = random.randint(0, NUM_LANES - 1)
        x = LANE_X[lane_idx] - OBSTACLE_WIDTH // 2
        y = -OBSTACLE_HEIGHT
        self.rect = pygame.Rect(x, y, OBSTACLE_WIDTH, OBSTACLE_HEIGHT)

    def update(self):
        self.rect.y += OBSTACLE_SPEED

    def draw(self, surface):
        pygame.draw.rect(surface, OBSTACLE_COLOR, self.rect)

    def off_screen(self):
        return self.rect.top > HEIGHT


def draw_lanes(surface):
    for i in range(1, NUM_LANES):
        x = i * LANE_WIDTH
        pygame.draw.line(surface, LANE_COLOR, (x, 0), (x, HEIGHT), 4)


def draw_text_center(surface, text, font, color, y):
    img = font.render(text, True, color)
    rect = img.get_rect(center=(WIDTH // 2, y))
    surface.blit(img, rect)


def game_loop():
    player = Player()
    obstacles = []
    last_spawn_time = pygame.time.get_ticks()
    start_time = pygame.time.get_ticks()
    score = 0

    running = True
    game_over = False

    while running:
        OBSTACLE_SPEED = 7 + score * 0.5
        dt = clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if not game_over:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        player.move_left()
                    elif event.key == pygame.K_RIGHT:
                        player.move_right()
            else:
                # On game over, any key press restarts
                if event.type == pygame.KEYDOWN:
                    return  # return to main() to restart

        if not game_over:
            # --- Spawn obstacles ---
            current_time = pygame.time.get_ticks()
            if current_time - last_spawn_time > OBSTACLE_SPAWN_TIME:
                obstacles.append(Obstacle())
                last_spawn_time = current_time

            # --- Update obstacles ---
            for obs in obstacles:
                obs.rect.y += int(OBSTACLE_SPEED)

            # Remove off-screen obstacles
            obstacles = [o for o in obstacles if not o.off_screen()]

            # --- Collision detection ---
            for obs in obstacles:
                if player.rect.colliderect(obs.rect):
                    game_over = True
                    score = (pygame.time.get_ticks() - start_time) // 1000
                    break

            # Update score (seconds survived)
            if not game_over:
                score = (pygame.time.get_ticks() - start_time) // 1000


        screen.fill(BG_COLOR)
        draw_lanes(screen)

        for obs in obstacles:
            obs.draw(screen)

        player.draw(screen)

        score_text = font.render(f"Score: {score}", True, TEXT_COLOR)
        screen.blit(score_text, (10, 10))

        if game_over:
            draw_text_center(screen, "GAME OVER", big_font, TEXT_COLOR, HEIGHT // 2 - 40)
            draw_text_center(screen, f"Score: {score}", font, TEXT_COLOR, HEIGHT // 2 + 10)
            draw_text_center(screen, "Press any key to play again", font, TEXT_COLOR, HEIGHT // 2 + 60)

        pygame.display.flip()


def main():
    while True:
        game_loop()

if __name__ == "__main__":
    main()