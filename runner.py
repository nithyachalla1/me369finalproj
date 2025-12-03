import pygame
import sys
import random

# ----- CONFIG -----
width, height = 480, 720
fps = 60

num_lanes = 3
lane_width = width // num_lanes

player_width = 50
player_height = 80
player_y = height - player_height - 30

obstacle_width = 50
obstacle_height = 80
obstacle_speed = 7
obstacle_spawn_time = 900

bg_color = (10, 10, 25)
lane_color = (60, 60, 90)
player_color = (50, 220, 70)
'''red_unlocked = False
blue_unlocked = False'''
obstacle_color = (220, 50, 50)
text_color = (255, 255, 255)

pygame.init()
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Three-Lane Runner")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 36)
big_font = pygame.font.SysFont(None, 72)
coin_img = pygame.image.load("coin.png").convert_alpha()
#coin_img = pygame.transform.scale(coin_img, (50, 50))


lane_x = []
for i in range(num_lanes):
    center_x = i * lane_width + lane_width // 2
    lane_x.append(center_x)


class Player:
    def __init__(self):
        self.lane_idx = 1
        self.rect = pygame.Rect(0, 0, player_width, player_height)
        self.update_position()

    def update_position(self):
        self.rect.centerx = lane_x[self.lane_idx]
        self.rect.y = player_y

    def move_left(self):
        if self.lane_idx > 0:
            self.lane_idx -= 1
            self.update_position()

    def move_right(self):
        if self.lane_idx < num_lanes - 1:
            self.lane_idx += 1
            self.update_position()

    def draw(self, surface):
        pygame.draw.rect(surface, player_color, self.rect)


class Obstacle:
    def __init__(self):
        lane_idx = random.randint(0, num_lanes - 1)
        x = lane_x[lane_idx] - obstacle_width // 2
        y = -obstacle_height
        self.rect = pygame.Rect(x, y, obstacle_width, obstacle_height)

    def update(self, speed):
        self.rect.y += speed

    def draw(self, surface):
        pygame.draw.rect(surface, obstacle_color, self.rect)

    def off_screen(self):
        return self.rect.top > height

class Coin:
    def __init__(self):
        lane_idx = random.randint(0, num_lanes - 1)
        x = lane_x[lane_idx] - 32  # center coin
        y = -30
        self.rect = pygame.Rect(x, y, 50, 50)

    def update(self, speed):
        self.rect.y += speed

    def draw(self, surface):
        surface.blit(coin_img, self.rect)

    def off_screen(self):
        return self.rect.top > height


def draw_lanes(surface):
    for i in range(1, num_lanes):
        x = i * lane_width
        pygame.draw.line(surface, lane_color, (x, 0), (x, height), 4)


def draw_text_center(surface, text, font, color, y):
    img = font.render(text, True, color)
    rect = img.get_rect(center=(width // 2, y))
    surface.blit(img, rect)


def game_loop(total_coins):
    player = Player()
    obstacles = []
    last_spawn_time = pygame.time.get_ticks()
    start_time = pygame.time.get_ticks()
    score = 0
    coins = []
    last_coin_spawn = pygame.time.get_ticks()
    coins_collected = total_coins
    coin_spawn_time = 1000

    running = True
    game_over = False

    while running:
        obstacle_speed_dynamic = 7 + score * 0.5
        dt = clock.tick(fps)

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
                '''if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_s:
                        return shop_loop(coins_collected)'''
                if event.type == pygame.KEYDOWN:
                    return coins_collected

        if not game_over:
            # Spawn obstacles and coins
            current_time = pygame.time.get_ticks()
            if current_time - last_spawn_time > obstacle_spawn_time:
                obstacles.append(Obstacle())
                last_spawn_time = current_time

            if current_time - last_coin_spawn > coin_spawn_time:
                candidate = Coin()
                # check for overlap
                if not any(candidate.rect.colliderect(o.rect) for o in obstacles):
                    coins.append(candidate)
                last_coin_spawn = current_time

            # Update obstacles and coins
            for obs in obstacles:
                obs.update(int(obstacle_speed_dynamic))

            for coin in coins:
                coin.update(int(obstacle_speed_dynamic))

            # Remove off-screen
            obstacles = [o for o in obstacles if not o.off_screen()]
            coins = [c for c in coins if not c.off_screen()]

            # Collision detection
            for obs in obstacles:
                if player.rect.colliderect(obs.rect):
                    game_over = True
                    score = (pygame.time.get_ticks() - start_time) // 1000
                    break
            # coin collection
            new_coins = []
            for coin in coins:
                if player.rect.colliderect(coin.rect):
                    coins_collected += 1
                else:
                    new_coins.append(coin)

            coins = new_coins

            # Score updates
            if not game_over:
                score = (pygame.time.get_ticks() - start_time) // 1000

        # draw
        screen.fill(bg_color)
        draw_lanes(screen)

        for obs in obstacles:
            obs.draw(screen)

        for coin in coins:
            coin.draw(screen)

        player.draw(screen)

        score_text = font.render(f"Score: {score}", True, text_color)
        screen.blit(score_text, (10, 10))
        coins_text = font.render(f"Coins: {coins_collected}", True, text_color)
        screen.blit(coins_text, (width - 150, 10))

        if game_over:
            draw_text_center(screen, "GAME OVER", big_font, text_color, height // 2 - 40)
            draw_text_center(screen, f"Score: {score}", font, text_color, height // 2 + 10)
            draw_text_center(screen, "Press any key to play again", font, text_color, height // 2 + 60)

        pygame.display.flip()


'''def shop_loop(total_coins):
    """
    Simple shop where you can spend coins to change the player color.
    Press:
      1 = default green (free)
      2 = red skin (cost 5)
      3 = blue skin (cost 5)
      ENTER/SPACE = exit shop and start next run
    """
    global player_color
    global red_unlocked
    global blue_unlocked

    running = True
    message = ""

    while running:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    # default skin
                    player_color = (50, 220, 70)
                    message = "Equipped: Default (free)"
                elif event.key == pygame.K_2:
                    # red skin, costs 5 coins
                    cost = 5
                    if not red_unlocked:
                        if total_coins >= cost:
                            total_coins -= cost
                            red_unlocked = True
                            player_color = (220, 60, 60)
                            message = "Bought & equipped: Red (-5 coins)"
                        else:
                            message = "Not enough coins for Red (5)"
                    else:
                        message = "Red already unlocked"
                elif event.key == pygame.K_3:
                    # blue skin, costs 5 coins
                    cost = 5
                    if total_coins >= cost:
                        total_coins -= cost
                        player_color = (60, 120, 255)
                        message = "Bought & equipped: Blue (-5 coins)"
                    else:
                        message = "Not enough coins for Blue (5)"
                elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    running = False

        # draw screen
        screen.fill((20, 20, 40))

        title_text = big_font.render("SHOP", True, text_color)
        title_rect = title_text.get_rect(center=(width // 2, 100))
        screen.blit(title_text, title_rect)

        coins_text = font.render(f"Coins: {total_coins}", True, text_color)
        screen.blit(coins_text, (20, 20))

        # Options
        line_y = 200
        line_spacing = 40

        option1 = font.render("1 - Default Green (free)", True, text_color)
        screen.blit(option1, (80, line_y))

        option2 = font.render("2 - Red Skin (5 coins)", True, text_color)
        screen.blit(option2, (80, line_y + line_spacing))

        option3 = font.render("3 - Blue Skin (5 coins)", True, text_color)
        screen.blit(option3, (80, line_y + 2 * line_spacing))

        exit_msg = font.render("Press ENTER/SPACE to start next run", True, text_color)
        exit_rect = exit_msg.get_rect(center=(width // 2, height - 80))
        screen.blit(exit_msg, exit_rect)

        if message:
            msg_text = font.render(message, True, text_color)
            msg_rect = msg_text.get_rect(center=(width // 2, height // 2 + 80))
            screen.blit(msg_text, msg_rect)

        pygame.display.flip()

    return total_coins'''



def main():
    total_coins = 0
    while True:
        total_coins = game_loop(total_coins)

if __name__ == "__main__":
    main()
