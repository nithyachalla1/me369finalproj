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


class MovementDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
            model_complexity=0
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.center_x = None
        self.center_y = None
        self.baseline_set = False
        self.movement_threshold = 0.05
        self.current_movement = 'center'
        self.running = True
        self.last_lane_change = 0
        self.lane_change_cooldown = 200
        
    def set_baseline(self, x, y):
        self.center_x = x
        self.center_y = y
        self.baseline_set = True
        
    def detect_movement(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        movement = 'center'
        h, w, _ = frame.shape
        current_time = pygame.time.get_ticks()
        
        if results.pose_landmarks:
            left_shoulder = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            
            current_x = (left_shoulder.x + right_shoulder.x) / 2
            current_y = (left_shoulder.y + right_shoulder.y) / 2
            
            if not self.baseline_set:
                self.set_baseline(current_x, current_y)
            
            x_threshold = self.movement_threshold
            x_diff = current_x - self.center_x
            
            if current_time - self.last_lane_change > self.lane_change_cooldown:
                if x_diff < -x_threshold:
                    movement = 'left'
                    self.last_lane_change = current_time
                elif x_diff > x_threshold:
                    movement = 'right'
                    self.last_lane_change = current_time
                else:
                    movement = 'center'
            else:
                movement = self.current_movement
            
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
            
            cv2.putText(frame, f"Movement: {movement.upper()}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No person detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            movement = 'center'
            
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


def draw_lanes(surface):
    for i in range(1, NUM_LANES):
        x = i * LANE_WIDTH
        pygame.draw.line(surface, LANE_COLOR, (x, 0), (x, HEIGHT), 4)


def draw_text_center(surface, text, font, color, y):
    img = font.render(text, True, color)
    rect = img.get_rect(center=(WIDTH // 2, y))
    surface.blit(img, rect)


def car_selection_screen(car_options):
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Choose Your Car")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 48)
    small_font = pygame.font.SysFont(None, 32)
    
    selected_idx = 0
    car_names = ["Green Car", "Blue Car", "Purple Car", "White/Red Car"]
    
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
        
        screen.fill(BG_COLOR)
        
        draw_text_center(screen, "SELECT YOUR CAR", font, TEXT_COLOR, 100)
        draw_text_center(screen, "Use LEFT/RIGHT arrows to choose", small_font, (200, 200, 200), 160)
        draw_text_center(screen, "Press ENTER to start", small_font, (200, 200, 200), 200)
        
        car_y = HEIGHT // 2 - 50
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
        
        cv2.imshow('Body Control - Runner Game', annotated_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            detector.running = False
            pygame.event.post(pygame.event.Event(pygame.QUIT))
            break
        elif key == ord('c'):
            detector.reset_baseline()
            print("Calibrating... Stand in center position")


def game_loop(detector, player_img, obstacle_imgs, bg_img, bg_y_offset):
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Body-Controlled Racing Game")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 36)
    big_font = pygame.font.SysFont(None, 72)
    
    try:
        pygame.mixer.music.load("Royalty Free Retro Gaming Music - Racing.mp3")
        pygame.mixer.music.set_volume(0.3)
        pygame.mixer.music.play(-1)
    except:
        print("Could not load music file")
    
    player = Player(player_img)
    obstacles = []
    last_spawn_time = pygame.time.get_ticks()
    start_time = pygame.time.get_ticks()
    score = 0

    running = True
    game_over = False

    while running and detector.running:
        current_speed = 7 + score * 0.5
        dt = clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                detector.stop()

            if not game_over:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        player.move_left()
                    elif event.key == pygame.K_RIGHT:
                        player.move_right()
            else:
                if event.type == pygame.KEYDOWN:
                    return True

        if not game_over:
            movement = detector.current_movement
            if movement == 'left':
                player.move_left()
            elif movement == 'right':
                player.move_right()

            current_time = pygame.time.get_ticks()
            if current_time - last_spawn_time > OBSTACLE_SPAWN_TIME:
                obstacles.append(Obstacle(obstacle_imgs))
                last_spawn_time = current_time

            for obs in obstacles:
                obs.update(current_speed)

            obstacles = [o for o in obstacles if not o.off_screen()]

            for obs in obstacles:
                if player.rect.colliderect(obs.rect):
                    game_over = True
                    score = (pygame.time.get_ticks() - start_time) // 1000
                    break

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

        player.draw(screen)

        score_text = font.render(f"Score: {score}", True, TEXT_COLOR)
        screen.blit(score_text, (10, 10))
        
        if not detector.baseline_set:
            calib_text = font.render("Press 'C' to calibrate", True, (255, 255, 0))
            screen.blit(calib_text, (10, 50))

        if game_over:
            draw_text_center(screen, "GAME OVER", big_font, TEXT_COLOR, HEIGHT // 2 - 40)
            draw_text_center(screen, f"Score: {score}", font, TEXT_COLOR, HEIGHT // 2 + 10)
            draw_text_center(screen, "Press any key to play again", font, TEXT_COLOR, HEIGHT // 2 + 60)

        pygame.display.flip()

    return False


def main():
    pygame.init()
    
    print("=" * 50)
    print("Body-Controlled Racing Game")
    print("=" * 50)
    print("\nControls:")
    print("1. Stand in front of your camera")
    print("2. Press 'C' in the camera window to calibrate")
    print("3. Move your body LEFT/RIGHT to control the car")
    print("4. Arrow keys work as backup controls")
    print("5. Press 'Q' to quit\n")
    print("Loading game assets...\n")
    
    print(f"Current working directory: {os.getcwd()}")
    sys.stdout.flush()
    print(f"Files in directory: {[f for f in os.listdir('.') if f.endswith('.png')]}")
    sys.stdout.flush()
    
    car_options = []
    player_car_files = [
        "Green Race car (front).png",
        "blue Race car (front).png",
        "Purple Race car (front).png",
        "WR Race car (back).png"
    ]
    
    for filename in player_car_files:
        img = load_and_scale_image(filename, PLAYER_WIDTH, PLAYER_HEIGHT)
        car_options.append(img)
    
    player_img = car_selection_screen(car_options)
    
    if player_img is None:
        print("Warning: Could not load player car, using colored rectangle")
    
    obstacle_imgs = []
    obstacle_files = [
        "police Race car (front).png"
    ]
    for filename in obstacle_files:
        img = load_and_scale_image(filename, OBSTACLE_WIDTH, OBSTACLE_HEIGHT)
        if img:
            obstacle_imgs.append(img)
        else:
            print(f"Warning: Could not load {filename}, using colored rectangle")
    
    if not obstacle_imgs:
        obstacle_imgs = None
    
    bg_img = None
    try:
        print("Attempting to load: Road Background 1.png")
        bg_img = pygame.image.load("Road Background 1.png")
        print("Successfully loaded background")
        bg_img = pygame.transform.scale(bg_img, (WIDTH, HEIGHT))
    except Exception as e:
        print(f"Error loading background: {e}")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    detector = MovementDetector()
    
    cam_thread = threading.Thread(target=camera_thread, args=(detector, cap))
    cam_thread.daemon = True
    cam_thread.start()
    
    time.sleep(1)
    
    bg_y_offset = [0]
    
    restart = True
    while restart and detector.running:
        restart = game_loop(detector, player_img, obstacle_imgs, bg_img, bg_y_offset)
    
    detector.stop()
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()
    print("Game closed. Thanks for playing!")


if __name__ == "__main__":
    main()