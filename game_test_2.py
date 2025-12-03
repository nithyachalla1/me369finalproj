import pygame
import sys
import random
import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import os

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

COIN_SIZE = 50
COIN_SPAWN_TIME = 1000

BG_COLOR = (10, 10, 25)
LANE_COLOR = (60, 60, 90)
TEXT_COLOR = (255, 255, 255)

LANE_X = []
for i in range(NUM_LANES):
    center_x = i * LANE_WIDTH + LANE_WIDTH // 2
    LANE_X.append(center_x)


def load_and_scale_image(filename, width, height):
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        print(f"Current directory: {os.getcwd()}")
        return None
    
    try:
        print(f"Attempting to load: {filename}")
        img = pygame.image.load(filename)
        print(f"Successfully loaded: {filename}")
        return pygame.transform.scale(img, (width, height))
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None


class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_hands=1
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.center_x = None
        self.center_y = None
        self.baseline_set = False
        self.movement_threshold = 0.08
        self.current_movement = 'center'
        self.current_gesture = None
        self.running = True
        self.last_lane_change = 0
        self.lane_change_cooldown = 200
    
    def detect_gesture(self, hand_landmarks):
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
        
        index_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
        middle_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        ring_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_MCP]
        pinky_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_MCP]
        
        fingers_closed = (
            index_tip.y > index_mcp.y and
            middle_tip.y > middle_mcp.y and
            ring_tip.y > ring_mcp.y and
            pinky_tip.y > pinky_mcp.y
        )
        
        if fingers_closed:
            return 'fist'
        
        thumb_index_dist = abs(thumb_tip.x - index_tip.x) + abs(thumb_tip.y - index_tip.y)
        if thumb_index_dist < 0.05:
            return 'pinch'
        
        index_up = index_tip.y < index_mcp.y
        middle_up = middle_tip.y < middle_mcp.y
        middle_down = middle_tip.y > middle_mcp.y
        ring_down = ring_tip.y > ring_mcp.y
        pinky_down = pinky_tip.y > pinky_mcp.y
        
        if index_up and middle_up and ring_down and pinky_down:
            return 'peace'
        
        if index_up and middle_down and ring_down and pinky_down:
            return 'point'
        
        return 'open'
        
    def set_baseline(self, x, y):
        self.center_x = x
        self.center_y = y
        self.baseline_set = True
        
    def detect_movement(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        movement = 'center'
        gesture = None
        h, w, _ = frame.shape
        current_time = pygame.time.get_ticks()
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            gesture = self.detect_gesture(hand_landmarks)
            self.current_gesture = gesture
            
            wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
            middle_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
            
            current_x = (wrist.x + middle_mcp.x) / 2
            current_y = (wrist.y + middle_mcp.y) / 2
            
            if not self.baseline_set:
                self.set_baseline(current_x, current_y)
            
            x_threshold = self.movement_threshold
            
            x_diff = current_x - self.center_x
            
            if x_diff < -x_threshold:
                movement = 'left'
            elif x_diff > x_threshold:
                movement = 'right'
            else:
                movement = 'center'
            
            self.mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS
            )
            
            if self.baseline_set:
                box_left = int((self.center_x - x_threshold) * w)
                box_right = int((self.center_x + x_threshold) * w)
                box_top = int(0.2 * h)
                box_bottom = int(0.8 * h)
                
                cv2.rectangle(frame, (box_left, box_top), (box_right, box_bottom), (0, 255, 0), 2)
                
                center_pixel = (int(self.center_x * w), int(self.center_y * h))
                cv2.circle(frame, center_pixel, 5, (0, 255, 0), -1)
                
                current_pixel = (int(current_x * w), int(current_y * h))
                cv2.circle(frame, current_pixel, 5, (0, 0, 255), -1)
            
            cv2.putText(frame, f"Lane: {movement.upper()}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Gesture: {gesture.upper()}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        else:
            cv2.putText(frame, "No hand detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            movement = 'center'
            gesture = None
            self.current_gesture = None
            
        self.current_movement = movement
        return movement, frame
    
    def reset_baseline(self):
        self.baseline_set = False
        self.center_x = None
        self.center_y = None
    
    def stop(self):
        self.running = False


class Player:
    def __init__(self, image):
        self.lane_idx = 1
        self.image = image
        if self.image:
            self.rect = self.image.get_rect()
        else:
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
        if self.image:
            surface.blit(self.image, self.rect)
        else:
            pygame.draw.rect(surface, (50, 220, 70), self.rect)


class Obstacle:
    def __init__(self, images):
        lane_idx = random.randint(0, NUM_LANES - 1)
        self.image = random.choice(images) if images else None
        
        if self.image:
            self.rect = self.image.get_rect()
            self.rect.centerx = LANE_X[lane_idx]
            self.rect.y = -self.rect.height
        else:
            x = LANE_X[lane_idx] - OBSTACLE_WIDTH // 2
            y = -OBSTACLE_HEIGHT
            self.rect = pygame.Rect(x, y, OBSTACLE_WIDTH, OBSTACLE_HEIGHT)

    def update(self, speed):
        self.rect.y += int(speed)

    def draw(self, surface):
        if self.image:
            surface.blit(self.image, self.rect)
        else:
            pygame.draw.rect(surface, (220, 50, 50), self.rect)

    def off_screen(self):
        return self.rect.top > HEIGHT


class Coin:
    def __init__(self, image):
        lane_idx = random.randint(0, NUM_LANES - 1)
        self.image = image
        if self.image:
            self.rect = self.image.get_rect()
            self.rect.centerx = LANE_X[lane_idx]
            self.rect.y = -COIN_SIZE
        else:
            x = LANE_X[lane_idx] - COIN_SIZE // 2
            y = -COIN_SIZE
            self.rect = pygame.Rect(x, y, COIN_SIZE, COIN_SIZE)

    def update(self, speed):
        self.rect.y += int(speed)

    def draw(self, surface):
        if self.image:
            surface.blit(self.image, self.rect)
        else:
            pygame.draw.circle(surface, (255, 215, 0), self.rect.center, COIN_SIZE // 2)

    def off_screen(self):
        return self.rect.top > HEIGHT


def draw_lanes(surface):
    for i in range(1, NUM_LANES):
        x = i * LANE_WIDTH
        pygame.draw.line(surface, LANE_COLOR, (x, 0), (x, HEIGHT), 4)


def car_selection_screen(car_options, detector):
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Choose Your Car")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 48)
    small_font = pygame.font.SysFont(None, 32)
    tiny_font = pygame.font.SysFont(None, 24)
    
    selected_idx = 0
    car_names = ["Red", "White/Red", "Purple"]
    
    selecting = True
    while selecting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    selected_idx = (selected_idx - 1) % len(car_options)
                elif event.key == pygame.K_RIGHT:
                    selected_idx = (selected_idx + 1) % len(car_options)
                elif event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                    return car_options[selected_idx]
        
        gesture = detector.current_gesture
        if gesture == 'point':
            return car_options[selected_idx]
        elif gesture == 'open':
            movement = detector.current_movement
            if movement == 'left':
                selected_idx = (selected_idx - 1) % len(car_options)
                time.sleep(0.3)
            elif movement == 'right':
                selected_idx = (selected_idx + 1) % len(car_options)
                time.sleep(0.3)
        
        screen.fill(BG_COLOR)
        
        draw_text_center(screen, "SELECT YOUR CAR", font, TEXT_COLOR, 80)
        draw_text_center(screen, "Arrows or Open Hand to choose", small_font, (200, 200, 200), 140)
        draw_text_center(screen, "Point finger or ENTER to select", small_font, (200, 200, 200), 180)
        
        if gesture:
            gesture_text = tiny_font.render(f"Gesture: {gesture.upper()}", True, (255, 255, 0))
            screen.blit(gesture_text, (10, 10))
        
        car_y = HEIGHT // 2 - 20
        total_width = len(car_options) * 120
        start_x = (WIDTH - total_width) // 2 + 60
        
        for i, car_img in enumerate(car_options):
            x = start_x + i * 120
            
            if i == selected_idx:
                pygame.draw.rect(screen, (255, 255, 0), (x - 35, car_y - 10, 70, 100), 3)
            
            if car_img:
                car_rect = car_img.get_rect(center=(x, car_y + 40))
                screen.blit(car_img, car_rect)
            else:
                pygame.draw.rect(screen, (100, 100, 100), (x - 25, car_y, 50, 80))
            
            name_surf = small_font.render(car_names[i], True, TEXT_COLOR if i == selected_idx else (150, 150, 150))
            name_rect = name_surf.get_rect(center=(x, car_y + 110))
            screen.blit(name_surf, name_rect)
        
        pygame.display.flip()
        clock.tick(FPS)
    
    return car_options[0]


def camera_thread(detector, cap):
    while detector.running and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        movement, annotated_frame = detector.detect_movement(frame)
        
        cv2.putText(annotated_frame, "Press 'C' to calibrate", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(annotated_frame, "Press 'Q' to quit", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.imshow('Hand Control - Runner Game', annotated_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            detector.running = False
            pygame.event.post(pygame.event.Event(pygame.QUIT))
            break
        elif key == ord('c'):
            detector.reset_baseline()
            print("Calibrating... Show your hand in center position")


def draw_text_center(surface, text, font, color, y):
    img = font.render(text, True, color)
    rect = img.get_rect(center=(WIDTH // 2, y))
    surface.blit(img, rect)


def game_loop(detector, player_img, obstacle_imgs, bg_img, bg_y_offset, coin_img):
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Hand-Controlled Racing Game")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 36)
    big_font = pygame.font.SysFont(None, 72)
    small_font = pygame.font.SysFont(None, 32)
    
    try:
        pygame.mixer.music.load("Royalty Free Retro Gaming Music - Racing.mp3")
        pygame.mixer.music.set_volume(0.3)
        pygame.mixer.music.play(-1)
    except:
        print("Could not load music file")
    
    player = Player(player_img)
    obstacles = []
    coins = []
    last_spawn_time = pygame.time.get_ticks()
    last_coin_spawn = pygame.time.get_ticks()
    start_time = pygame.time.get_ticks()
    score = 0
    coins_collected = 0
    pause_start_time = 0

    running = True
    game_over = False
    paused = False
    last_fist_time = 0
    fist_cooldown = 500

    while running and detector.running:
        current_speed = 7 + score * 0.5
        dt = clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                detector.stop()

            if not game_over and not paused:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        player.move_left()
                    elif event.key == pygame.K_RIGHT:
                        player.move_right()
                    elif event.key == pygame.K_SPACE:
                        paused = True
                        pause_start_time = pygame.time.get_ticks()
            elif paused:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        paused = False
                        time_paused = pygame.time.get_ticks() - pause_start_time
                        start_time += time_paused
                        last_spawn_time += time_paused
            else:
                if event.type == pygame.KEYDOWN:
                    return True
        
        gesture = detector.current_gesture
        current_time = pygame.time.get_ticks()
        
        if gesture == 'fist':
            running = False
            detector.stop()
            return False
        
        if not game_over and gesture == 'peace' and current_time - last_fist_time > fist_cooldown:
            paused = not paused
            if paused:
                pause_start_time = current_time
            else:
                time_paused = current_time - pause_start_time
                start_time += time_paused
                last_spawn_time += time_paused
            last_fist_time = current_time
        
        if game_over and gesture == 'pinch':
            return True

        if not game_over and not paused:
            
            movement = detector.current_movement
            
            if movement == 'left' and player.lane_idx != 0:
                player.lane_idx = 0
                player.update_position()
            elif movement == 'right' and player.lane_idx != 2:
                player.lane_idx = 2
                player.update_position()
            elif movement == 'center' and player.lane_idx != 1:
                player.lane_idx = 1
                player.update_position()

            current_time = pygame.time.get_ticks()
            if current_time - last_spawn_time > OBSTACLE_SPAWN_TIME:
                obstacles.append(Obstacle(obstacle_imgs))
                last_spawn_time = current_time
            
            if current_time - last_coin_spawn > COIN_SPAWN_TIME:
                candidate = Coin(coin_img)
                if not any(candidate.rect.colliderect(o.rect) for o in obstacles):
                    coins.append(candidate)
                last_coin_spawn = current_time

            for obs in obstacles:
                obs.update(current_speed)
            
            for coin in coins:
                coin.update(current_speed)

            obstacles = [o for o in obstacles if not o.off_screen()]
            coins = [c for c in coins if not c.off_screen()]

            for obs in obstacles:
                if player.rect.colliderect(obs.rect):
                    game_over = True
                    score = (pygame.time.get_ticks() - start_time) // 1000
                    break
            
            new_coins = []
            for coin in coins:
                if player.rect.colliderect(coin.rect):
                    coins_collected += 1
                else:
                    new_coins.append(coin)
            coins = new_coins

            if not game_over:
                score = (pygame.time.get_ticks() - start_time) // 1000

        if bg_img:
            bg_y = (bg_y_offset[0] % HEIGHT)
            screen.blit(bg_img, (0, bg_y - HEIGHT))
            screen.blit(bg_img, (0, bg_y))
            bg_y_offset[0] += current_speed / 2
        else:
            screen.fill(BG_COLOR)
        
        draw_lanes(screen)

        for obs in obstacles:
            obs.draw(screen)
        
        for coin in coins:
            coin.draw(screen)

        player.draw(screen)

        score_text = font.render(f"Score: {score}", True, TEXT_COLOR)
        screen.blit(score_text, (10, 10))
        
        coins_text = font.render(f"Coins: {coins_collected}", True, TEXT_COLOR)
        screen.blit(coins_text, (WIDTH - 150, 10))
        
        if paused:
            overlay = pygame.Surface((WIDTH, HEIGHT))
            overlay.set_alpha(128)
            overlay.fill((0, 0, 0))
            screen.blit(overlay, (0, 0))
            
            draw_text_center(screen, "PAUSED", big_font, TEXT_COLOR, HEIGHT // 2 - 40)
            draw_text_center(screen, "Peace Sign to resume or SPACE", font, TEXT_COLOR, HEIGHT // 2 + 20)
            draw_text_center(screen, "Fist to quit game", small_font, (255, 100, 100), HEIGHT // 2 + 60)
        
        if not detector.baseline_set and not paused and not game_over:
            calib_text = font.render("Press 'C' to calibrate", True, (255, 255, 0))
            screen.blit(calib_text, (10, 50))

        if game_over:
            draw_text_center(screen, "GAME OVER", big_font, TEXT_COLOR, HEIGHT // 2 - 60)
            draw_text_center(screen, f"Score: {score}", font, TEXT_COLOR, HEIGHT // 2)
            draw_text_center(screen, f"Coins: {coins_collected}", font, TEXT_COLOR, HEIGHT // 2 + 40)
            draw_text_center(screen, "Pinch to restart or press any key", small_font, TEXT_COLOR, HEIGHT // 2 + 90)

        pygame.display.flip()

    return False


def main():
    pygame.init()
    
    print("=" * 50)
    print("Hand-Controlled Racing Game")
    print("=" * 50)
    print("\nControls:")
    print("1. Show your hand in front of camera")
    print("2. Press 'C' in the camera window to calibrate")
    print("3. Move your hand LEFT/RIGHT/UP/DOWN to control")
    print("4. Arrow keys work as backup controls")
    print("5. Press 'Q' to quit\n")
    print("Loading game assets...\n")
    
    print(f"Current working directory: {os.getcwd()}")
    sys.stdout.flush()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    detector = HandDetector()
    
    cam_thread = threading.Thread(target=camera_thread, args=(detector, cap))
    cam_thread.daemon = True
    cam_thread.start()
    
    time.sleep(1)
    
    car_options = []
    player_car_files = [
        "Player Car Red Scaled.png",
        "WR Player Car  Scaled.png",
        "Purple Race car (front).png"
    ]
    
    for filename in player_car_files:
        img = load_and_scale_image(filename, PLAYER_WIDTH, PLAYER_HEIGHT)
        car_options.append(img)
    
    player_img = car_selection_screen(car_options, detector)
    
    if player_img is None:
        print("Warning: Could not load player car, using colored rectangle")
    
    obstacle_imgs = []
    police_car = load_and_scale_image("Enemy Police Race Car Scaled.png", OBSTACLE_WIDTH, OBSTACLE_HEIGHT)
    if police_car:
        obstacle_imgs.append(police_car)
    else:
        print("Warning: Could not load police car, using colored rectangle")
    
    if not obstacle_imgs:
        obstacle_imgs = None
    
    coin_img = load_and_scale_image("Coin.png", COIN_SIZE, COIN_SIZE)
    if coin_img is None:
        print("Warning: Could not load coin image, using yellow circle")
    
    bg_img = None
    try:
        bg_img = pygame.image.load("Fixed Road 1.png")
        print("Successfully loaded background")
        bg_img = pygame.transform.scale(bg_img, (WIDTH, HEIGHT))
    except Exception as e:
        print(f"Error loading background: {e}")
    
    bg_y_offset = [0]
    
    restart = True
    while restart and detector.running:
        restart = game_loop(detector, player_img, obstacle_imgs, bg_img, bg_y_offset, coin_img)
    
    detector.stop()
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()
    print("Game closed. Thanks for playing!")


if __name__ == "__main__":
    main()