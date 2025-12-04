import pygame
import sys
import random
import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import math
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

POWERUP_SIZE = 60
POWERUP_MIN_SPAWN_TIME = 10000
POWERUP_MAX_SPAWN_TIME = 25000
POWERUP_DURATION = 5000

BG_COLOR = (10, 10, 25)
LANE_COLOR = (60, 60, 90)
TEXT_COLOR = (255, 255, 255)

total_coins_global = 0
unlocked_skins = [0]
current_skin_index = 0

LANE_X = []
for i in range(NUM_LANES):
    center_x = i * LANE_WIDTH + LANE_WIDTH // 2
    LANE_X.append(center_x)


def thumb_up_check(thumb_tip, index_mcp):
    return thumb_tip.y < index_mcp.y - 0.1


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


def load_animation_frames(filename, width, height, frame_count):

    if not os.path.exists(filename):
        print(f"Animation file not found: {filename}")
        return None
    try:
        print(f"Attempting to load animation: {filename}")
        sprite_sheet = pygame.image.load(filename).convert_alpha()
        sheet_width = sprite_sheet.get_width()
        sheet_height = sprite_sheet.get_height()

        known_ranges_69 = [
            (11, 17), (41, 47), (71, 77), (100, 107), (128, 138), (158, 168),
            (186, 199), (215, 230), (244, 261), (273, 290), (304, 315),
            (334, 345), (364, 381), (394, 411), (429, 450), (459, 480),
            (490, 509), (523, 545), (555, 575), (584, 603), (617, 637),
            (645, 666), (674, 696), (705, 728), (736, 758), (766, 787),
            (795, 816), (824, 846), (854, 875), (883, 905), (913, 934),
            (942, 964), (972, 993), (1001, 1023), (1031, 1053), (1061, 1082),
            (1090, 1111), (1119, 1141), (1149, 1170), (1178, 1200),
            (1208, 1229), (1237, 1259), (1267, 1288), (1296, 1317),
            (1325, 1347), (1355, 1376), (1384, 1405), (1413, 1435),
            (1443, 1464), (1472, 1493), (1501, 1523), (1531, 1552),
            (1560, 1581), (1589, 1611), (1619, 1640), (1648, 1669),
            (1677, 1698), (1706, 1727), (1735, 1756), (1764, 1785),
            (1793, 1814), (1822, 1844), (1852, 1873), (1881, 1902),
            (1910, 1931), (1939, 1960), (1968, 1989)
        ]

        frames = []

        if (frame_count == 69) or (sheet_width == 2048 and sheet_height <= 40):
            ranges = known_ranges_69
            for (x1, x2) in ranges:
                x1_clamped = max(0, min(sheet_width - 1, x1))
                x2_clamped = max(0, min(sheet_width - 1, x2))
                w_rect = x2_clamped - x1_clamped + 1
                rect = pygame.Rect(x1_clamped, 0, w_rect, sheet_height)
                frame = sprite_sheet.subsurface(rect).copy()
                scaled = pygame.transform.scale(frame, (width, height))
                frames.append(scaled)
            print(f"Loaded {len(frames)} frames using known ranges from {filename}")
            return frames

        if frame_count > 0 and (sheet_width % frame_count == 0):
            frame_w = sheet_width // frame_count
            for i in range(frame_count):
                rect = pygame.Rect(i * frame_w, 0, frame_w, sheet_height)
                frame = sprite_sheet.subsurface(rect).copy()
                frames.append(pygame.transform.scale(frame, (width, height)))
            print(f"Loaded {len(frames)} equal-width frames from {filename}")
            return frames

        columns_nonempty = [False] * sheet_width
        for x in range(sheet_width):
            for y in range(sheet_height):
                if sprite_sheet.get_at((x, y))[3] != 0:
                    columns_nonempty[x] = True
                    break

        runs = []
        in_run = False
        run_start = 0
        for x, val in enumerate(columns_nonempty):
            if val and not in_run:
                in_run = True
                run_start = x
            elif not val and in_run:
                in_run = False
                runs.append((run_start, x - 1))
        if in_run:
            runs.append((run_start, sheet_width - 1))

        if runs:
            for (s, e) in runs:
                w_rect = e - s + 1
                rect = pygame.Rect(s, 0, w_rect, sheet_height)
                frame = sprite_sheet.subsurface(rect).copy()
                frames.append(pygame.transform.scale(frame, (width, height)))
            print(f"Auto-detected and loaded {len(frames)} frames from {filename}")
            return frames

        print(f"Could not detect frames automatically for {filename}")
        return None

    except Exception as e:
        print(f"Error loading animation {filename}: {e}")
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
        self.movement_threshold = 0.07
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
        thumb_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_CMC]
        
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
        index_down = index_tip.y > index_mcp.y
        middle_up = middle_tip.y < middle_mcp.y
        middle_down = middle_tip.y > middle_mcp.y
        ring_up = ring_tip.y < ring_mcp.y
        ring_down = ring_tip.y > ring_mcp.y
        pinky_up = pinky_tip.y < pinky_mcp.y
        pinky_down = pinky_tip.y > pinky_mcp.y
        thumb_extended = abs(thumb_tip.x - thumb_mcp.x) > 0.1
        
        if index_up and pinky_up and middle_down and ring_down:
            return 'horns'
        
        all_fingers_up = index_up and middle_up and ring_up and pinky_up
        if all_fingers_up and thumb_extended:
            return 'open'
        
        if index_up and middle_up and ring_down and pinky_down:
            return 'peace'
        
        if index_up and middle_down and ring_down and pinky_down and not thumb_extended:
            return 'point'
        
        if index_up and middle_up and ring_up and pinky_down and not thumb_extended:
            return 'three'
        
        if index_up and middle_up and ring_up and pinky_up and not thumb_extended:
            return 'four'
        
        if pinky_up and thumb_extended and index_down and middle_down and ring_down:
            return 'rad'
        
        if index_up  and thumb_extended and middle_down and ring_down and pinky_down:
            return 'l_hand'
        
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
    def __init__(self, image, flying_animation=None):
        self.lane_idx = 1
        self.image = image
        self.flying_animation = flying_animation
        self.animation_frame = 0
        self.animation_speed = 0.8
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

    def draw(self, surface, invincible=False, fly_offset=0):
        draw_rect = self.rect.copy()
        if invincible:
            draw_rect.y += fly_offset
            glow_surface = pygame.Surface((self.rect.width + 20, self.rect.height + 20), pygame.SRCALPHA)
            glow_color = (255, 215, 0, 100)
            pygame.draw.ellipse(glow_surface, glow_color, glow_surface.get_rect())
            surface.blit(glow_surface, (draw_rect.x - 10, draw_rect.y - 10))
            
            if self.flying_animation:
                dt = 16 
                self.animation_frame += self.animation_speed * dt/1000
                self.animation_frame %= len(self.flying_animation)

                frame_index = int(self.animation_frame)
                surface.blit(self.flying_animation[frame_index], draw_rect)
            elif self.image:
                surface.blit(self.image, draw_rect)
            else:
                pygame.draw.rect(surface, (50, 220, 70), draw_rect)
        else:
            if self.image:
                surface.blit(self.image, draw_rect)
            else:
                pygame.draw.rect(surface, (50, 220, 70), draw_rect)


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


class PowerUp:
    def __init__(self, image):
        lane_idx = random.randint(0, NUM_LANES - 1)
        self.image = image
        if self.image:
            self.rect = self.image.get_rect()
            self.rect.centerx = LANE_X[lane_idx]
            self.rect.y = -POWERUP_SIZE
        else:
            x = LANE_X[lane_idx] - POWERUP_SIZE // 2
            y = -POWERUP_SIZE
            self.rect = pygame.Rect(x, y, POWERUP_SIZE, POWERUP_SIZE)

    def update(self, speed):
        self.rect.y += int(speed)

    def draw(self, surface):
        if self.image:
            surface.blit(self.image, self.rect)
        else:
            pygame.draw.circle(surface, (255, 100, 255), self.rect.center, POWERUP_SIZE // 2)

    def off_screen(self):
        return self.rect.top > HEIGHT


def draw_lanes(surface):
    for i in range(1, NUM_LANES):
        x = i * LANE_WIDTH
        pygame.draw.line(surface, LANE_COLOR, (x, 0), (x, HEIGHT), 4)


def start_screen(detector):
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("I-35 Racing Simulator Game")
    clock = pygame.time.Clock()
    title_font = pygame.font.SysFont(None, 56)
    header_font = pygame.font.SysFont(None, 32)
    text_font = pygame.font.SysFont(None, 24)
    small_font = pygame.font.SysFont(None, 20)
    running = True
    last_gesture_time = 0
    gesture_cooldown = 500
    
    while running and detector.running:
        current_time = pygame.time.get_ticks()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                detector.stop()
                return False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    return True
        
        gesture = detector.current_gesture
        if gesture == 'l_hand' and current_time - last_gesture_time > gesture_cooldown:
            return True
        elif gesture == 'fist' and current_time - last_gesture_time > gesture_cooldown:
            detector.stop()
            return False
        
        screen.fill(BG_COLOR)
        title = title_font.render("I-35 Simulator", True, (100, 200, 255))
        title_rect = title.get_rect(center=(WIDTH // 2, 50))
        screen.blit(title, title_rect)
        
        y_offset = 110
        rules_header = header_font.render("GAME RULES:", True, (255, 255, 100))
        screen.blit(rules_header, (20, y_offset))
        y_offset += 40
        
        rules = [
            "• Dodge obstacles to survive",
            "• Collect coins for shop purchases",
            "• Grab power-ups for invincibility",
            "• Speed increases as you play",
        ]
        for rule in rules:
            rule_text = text_font.render(rule, True, TEXT_COLOR)
            screen.blit(rule_text, (30, y_offset))
            y_offset += 30
        
        y_offset += 20
        controls_header = header_font.render("HAND CONTROLS:", True, (255, 255, 100))
        screen.blit(controls_header, (20, y_offset))
        y_offset += 40
        
        controls = [
            ("Open Hand (move)", "Switch lanes"),
            ("Point Finger", "Select options"),
            ("Peace Sign ✌", "Pause game / Buy car 1"),
            ("3 Fingers", "Buy car 2"),
            ("Hook Em 🤘", "Open shop"),
            ("Pinch", "Restart after crash"),
            ("Rad 👍", "Car selection"),
            ("L Hand 👈", "Continue / Start"),
            ("Fist", "Quit game"),
        ]
        for gesture_name, action in controls:
            gesture_text = small_font.render(f"• {gesture_name}:", True, (150, 255, 150))
            action_text = small_font.render(action, True, (200, 200, 200))
            screen.blit(gesture_text, (30, y_offset))
            screen.blit(action_text, (220, y_offset))
            y_offset += 25
        
        y_offset = HEIGHT - 80
        start_text = header_font.render("L HAND TO START", True, (100, 255, 100))
        start_rect = start_text.get_rect(center=(WIDTH // 2, y_offset))
        alpha = int(127.5 * (1 + math.sin(current_time * 0.003)))
        start_text.set_alpha(alpha)
        screen.blit(start_text, start_rect)
        
        if gesture:
            gesture_display = text_font.render(f"Gesture: {gesture.upper()}", True, (255, 255, 0))
            screen.blit(gesture_display, (10, HEIGHT - 30))
        
        keyboard_hint = small_font.render("Or press ENTER", True, (150, 150, 150))
        keyboard_rect = keyboard_hint.get_rect(center=(WIDTH // 2, HEIGHT - 40))
        screen.blit(keyboard_hint, keyboard_rect)
        pygame.display.flip()
        clock.tick(FPS)
    return False


def car_selection_screen(car_options, flying_animation, detector):
    global current_skin_index, unlocked_skins
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Choose Your Car")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 48)
    small_font = pygame.font.SysFont(None, 32)
    tiny_font = pygame.font.SysFont(None, 24)
    available_cars = [car_options[i] for i in unlocked_skins]
    car_names = ["Red Car", "Blue Car", "White Red Car, Police Car"]
    available_names = [car_names[i] for i in unlocked_skins]
    selected_idx = unlocked_skins.index(current_skin_index) if current_skin_index in unlocked_skins else 0
    selecting = True
    last_gesture_time = 0
    gesture_cooldown = 500
    
    while selecting:
        current_time = pygame.time.get_ticks()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    selected_idx = (selected_idx - 1) % len(available_cars)
                elif event.key == pygame.K_RIGHT:
                    selected_idx = (selected_idx + 1) % len(available_cars)
                elif event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                    current_skin_index = unlocked_skins[selected_idx]
                    return available_cars[selected_idx]
        
        gesture = detector.current_gesture
        if gesture and current_time - last_gesture_time > gesture_cooldown:
            if gesture == 'point':
                current_skin_index = unlocked_skins[selected_idx]
                return available_cars[selected_idx]
            elif gesture == 'open':
                movement = detector.current_movement
                if movement == 'left':
                    selected_idx = (selected_idx - 1) % len(available_cars)
                    last_gesture_time = current_time
                elif movement == 'right':
                    selected_idx = (selected_idx + 1) % len(available_cars)
                    last_gesture_time = current_time
        
        screen.fill(BG_COLOR)
        draw_text_center(screen, "SELECT YOUR CAR", font, TEXT_COLOR, 80)
        draw_text_center(screen, "Arrows or Open Hand to choose", small_font, (200, 200, 200), 140)
        draw_text_center(screen, "Point finger or ENTER to select", small_font, (200, 200, 200), 180)
        
        if gesture:
            gesture_text = tiny_font.render(f"Gesture: {gesture.upper()}", True, (255, 255, 0))
            screen.blit(gesture_text, (10, 10))
        
        car_y = HEIGHT // 2 - 20
        total_width = len(available_cars) * 120
        start_x = (WIDTH - total_width) // 2 + 60
        
        for i, car_img in enumerate(available_cars):
            x = start_x + i * 120
            if i == selected_idx:
                pygame.draw.rect(screen, (255, 255, 0), (x - 35, car_y - 10, 70, 100), 3)
            if car_img:
                car_rect = car_img.get_rect(center=(x, car_y + 40))
                screen.blit(car_img, car_rect)
            else:
                pygame.draw.rect(screen, (100, 100, 100), (x - 25, car_y, 50, 80))
            name_surf = small_font.render(available_names[i], True, TEXT_COLOR if i == selected_idx else (150, 150, 150))
            name_rect = name_surf.get_rect(center=(x, car_y + 110))
            screen.blit(name_surf, name_rect)
        pygame.display.flip()
        clock.tick(FPS)
    return available_cars[0]


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


def shop_loop(detector):
    global total_coins_global, unlocked_skins, current_skin_index
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Shop")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 36)
    big_font = pygame.font.SysFont(None, 72)
    small_font = pygame.font.SysFont(None, 28)
    running = True
    message = ""
    last_gesture_time = 0
    gesture_cooldown = 500
    start_text = "Point to return"
    
    while running and detector.running:
        current_time = pygame.time.get_ticks()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                detector.stop()
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    cost = 10
                    if 1 not in unlocked_skins:
                        if total_coins_global >= cost:
                            total_coins_global -= cost
                            unlocked_skins.append(1)
                            message = "Bought: Blue Car (-10 coins)"
                        else:
                            message = "Not enough coins for Blue (10)"
                    else:
                        message = "Already owned: Blue Car"
                elif event.key == pygame.K_2:
                    cost = 50
                    if 2 not in unlocked_skins:
                        if total_coins_global >= cost:
                            total_coins_global -= cost
                            unlocked_skins.append(2)
                            message = "Bought: White Red Car (-50 coins)"
                        else:
                            message = "Not enough coins for White Red (50)"
                    else:
                        message = "Already owned: White Red Car"
                elif event.key == pygame.K_3:
                    cost = 100
                    if 3 not in unlocked_skins:
                        if total_coins_global >= cost:
                            total_coins_global -= cost
                            unlocked_skins.append(3)
                            message = "Bought: Police Car (-100 coins)"
                        else:
                            message = "Not enough coins for Police Car (100)"
                    else:
                        message = "Already owned: Police Car"
                elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    running = False
        
        gesture = detector.current_gesture
        if gesture and current_time - last_gesture_time > gesture_cooldown:
            if gesture == 'point':
                running = False
            elif gesture == 'peace':
                cost = 10
                if 1 not in unlocked_skins:
                    if total_coins_global >= cost:
                        total_coins_global -= cost
                        unlocked_skins.append(1)
                        message = "Bought: Blue Car (-10 coins)"
                    else:
                        message = "Not enough coins for Blue (10)"
                else:
                    message = "Already owned: Blue Car"
                last_gesture_time = current_time
            elif gesture == 'three':
                cost = 50
                if 2 not in unlocked_skins:
                    if total_coins_global >= cost:
                        total_coins_global -= cost
                        unlocked_skins.append(2)
                        message = "Bought: White Red Car (-50 coins)"
                    else:
                        message = "Not enough coins for White Red (50)"
                else:
                    message = "Already owned: White Red Car"
                last_gesture_time = current_time
            elif gesture == 'four':
                cost = 100
                if 3 not in unlocked_skins:
                    if total_coins_global >= cost:
                        total_coins_global -= cost
                        unlocked_skins.append(3)
                        message = "Bought: Police Car (-100 coins)"
                    else:
                        message = "Not enough coins for Police (100)"
                else:
                    message = "Already owned: Police Car"
            elif gesture == 'fist':
                detector.stop()
                return False
        
        screen.fill((20, 20, 40))
        draw_text_center(screen, "SHOP", big_font, TEXT_COLOR, 80)
        coins_text = font.render(f"Coins: {total_coins_global}", True, (255, 215, 0))
        screen.blit(coins_text, (20, 20))
        line_y = 200
        line_spacing = 50
        
        blue_status = " [OWNED]" if 1 in unlocked_skins else " (10 coins)"
        option1_color = (100, 150, 255) if 1 in unlocked_skins else TEXT_COLOR
        option1 = font.render(f"Peace Sign - Blue Car{blue_status}", True, option1_color)
        screen.blit(option1, (60, line_y))
        
        wr_status = " [OWNED]" if 2 in unlocked_skins else " (50 coins)"
        option2_color = (200, 100, 255) if 2 in unlocked_skins else TEXT_COLOR
        option2 = font.render(f"3 Fingers - White Red Car{wr_status}", True, option2_color)
        screen.blit(option2, (60, line_y + line_spacing))

        police_status = " [OWNED]" if 3 in unlocked_skins else " (100 coins)"
        option3_color = (300, 50, 255) if 3 in unlocked_skins else TEXT_COLOR
        option3 = font.render(f"3 Fingers - Police Car{police_status}", True, option3_color)
        screen.blit(option3, (60, line_y + line_spacing))
        
        controls_y = HEIGHT - 150
        draw_text_center(screen, start_text, small_font, (100, 255, 100), controls_y)
        draw_text_center(screen, "Fist to quit", small_font, (255, 100, 100), controls_y + 35)
        
        if message:
            msg_text = font.render(message, True, (255, 255, 100))
            msg_rect = msg_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 100))
            screen.blit(msg_text, msg_rect)
        
        if gesture:
            gesture_text = small_font.render(f"Gesture: {gesture.upper()}", True, (255, 255, 0))
            screen.blit(gesture_text, (WIDTH - 200, 20))
        pygame.display.flip()
        clock.tick(FPS)
    return True


def game_loop(detector, player_img, flying_animation, obstacle_imgs, bg_img, bg_y_offset, coin_img, powerup_img, car_options):
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
    
    player = Player(player_img, flying_animation)
    obstacles = []
    coins = []
    powerups = []
    last_spawn_time = pygame.time.get_ticks()
    last_coin_spawn = pygame.time.get_ticks()
    last_powerup_spawn = pygame.time.get_ticks()
    next_powerup_time = random.randint(POWERUP_MIN_SPAWN_TIME, POWERUP_MAX_SPAWN_TIME)
    start_time = pygame.time.get_ticks()
    score = 0
    coins_collected = 0
    pause_start_time = 0
    invincible = False
    invincible_end_time = 0
    fly_offset = 0
    fly_direction = 1
    running = True
    game_over = False
    paused = False
    last_gesture_time = 0
    gesture_cooldown = 500

    while running and detector.running:
        current_speed = 7 + score * 0.5
        dt = clock.tick(FPS)
        current_time = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                detector.stop()
                return False, 0
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
                        last_coin_spawn += time_paused
            else:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_c:
                        return "car_selection", coins_collected
                    else:
                        return "restart", coins_collected
        
        gesture = detector.current_gesture
        if gesture and current_time - last_gesture_time > gesture_cooldown:
            if gesture == 'fist':
                running = False
                detector.stop()
                return False, 0
            if not game_over and gesture == 'peace':
                paused = not paused
                if paused:
                    pause_start_time = current_time
                else:
                    time_paused = current_time - pause_start_time
                    start_time += time_paused
                    last_spawn_time += time_paused
                    last_coin_spawn += time_paused
                last_gesture_time = current_time
            if not game_over and not paused and gesture == 'horns':
                paused = True
                pause_start_time = current_time
                continue_game = shop_loop(detector)
                if not continue_game:
                    return False, coins_collected
                player_img = car_options[current_skin_index]
                player.image = player_img
                time_paused = pygame.time.get_ticks() - pause_start_time
                start_time += time_paused
                last_spawn_time += time_paused
                last_coin_spawn += time_paused
                last_powerup_spawn += time_paused
                paused = False
                last_gesture_time = current_time
            if game_over and gesture == 'horns':
                continue_game = shop_loop(detector)
                if not continue_game:
                    return False, coins_collected
                last_gesture_time = current_time
            if game_over and gesture == 'pinch':
                return "restart", coins_collected
            if game_over and gesture == 'rad':
                return "car_selection", coins_collected
            if gesture == 'l_hand':
                if game_over:
                    return "car_selection", coins_collected
                elif paused:
                    paused = False
                    time_paused = current_time - pause_start_time
                    start_time += time_paused
                    last_spawn_time += time_paused
                    last_coin_spawn += time_paused
                    last_gesture_time = current_time

        if not game_over and not paused:
            movement = detector.current_movement
            if current_time > invincible_end_time:
                invincible = False
            if invincible:
                fly_offset += fly_direction * 2
                if fly_offset > 20 or fly_offset < -20:
                    fly_direction *= -1
            else:
                fly_offset = 0
            
            if movement == 'left' and player.lane_idx != 0:
                player.lane_idx = 0
                player.update_position()
            elif movement == 'right' and player.lane_idx != 2:
                player.lane_idx = 2
                player.update_position()
            elif movement == 'center' and player.lane_idx != 1:
                player.lane_idx = 1
                player.update_position()

            if current_time - last_spawn_time > OBSTACLE_SPAWN_TIME:
                obstacles.append(Obstacle(obstacle_imgs))
                last_spawn_time = current_time
            if current_time - last_coin_spawn > COIN_SPAWN_TIME:
                candidate = Coin(coin_img)
                if not any(candidate.rect.colliderect(o.rect) for o in obstacles):
                    coins.append(candidate)
                last_coin_spawn = current_time
            if current_time - last_powerup_spawn > next_powerup_time:
                candidate = PowerUp(powerup_img)
                if not any(candidate.rect.colliderect(o.rect) for o in obstacles):
                    powerups.append(candidate)
                last_powerup_spawn = current_time
                next_powerup_time = random.randint(POWERUP_MIN_SPAWN_TIME, POWERUP_MAX_SPAWN_TIME)

            for obs in obstacles:
                obs.update(current_speed)
            for coin in coins:
                coin.update(current_speed)
            for powerup in powerups:
                powerup.update(current_speed)

            obstacles = [o for o in obstacles if not o.off_screen()]
            coins = [c for c in coins if not c.off_screen()]
            powerups = [p for p in powerups if not p.off_screen()]

            if not invincible:
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
            
            new_powerups = []
            for powerup in powerups:
                if player.rect.colliderect(powerup.rect):
                    invincible = True
                    invincible_end_time = current_time + POWERUP_DURATION
                else:
                    new_powerups.append(powerup)
            powerups = new_powerups

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
        for powerup in powerups:
            powerup.draw(screen)
        player.draw(screen, invincible, fly_offset)

        score_text = font.render(f"Score: {score}", True, TEXT_COLOR)
        screen.blit(score_text, (10, 10))
        coins_text = font.render(f"Coins: {coins_collected}", True, TEXT_COLOR)
        screen.blit(coins_text, (WIDTH - 150, 10))
        
        if invincible:
            time_left = max(0, (invincible_end_time - current_time) / 1000)
            invincible_text = font.render(f"INVINCIBLE: {time_left:.1f}s", True, (255, 215, 0))
            screen.blit(invincible_text, (WIDTH // 2 - 100, 50))
        
        if paused:
            overlay = pygame.Surface((WIDTH, HEIGHT))
            overlay.set_alpha(128)
            overlay.fill((0, 0, 0))
            screen.blit(overlay, (0, 0))
            draw_text_center(screen, "PAUSED", big_font, TEXT_COLOR, HEIGHT // 2 - 40)
            draw_text_center(screen, "Peace Sign or L Hand to resume", font, TEXT_COLOR, HEIGHT // 2 + 20)
            draw_text_center(screen, "Fist to quit game", small_font, (255, 100, 100), HEIGHT // 2 + 60)
        
        if not detector.baseline_set and not paused and not game_over:
            calib_text = font.render("Press 'C' to calibrate", True, (255, 255, 0))
            screen.blit(calib_text, (10, 50))
        
        shop_hint = small_font.render("Hook Em (horns) for Shop", True, (200, 200, 200))
        screen.blit(shop_hint, (10, HEIGHT - 40))

        if game_over:
            draw_text_center(screen, "GAME OVER", big_font, TEXT_COLOR, HEIGHT // 2 - 60)
            draw_text_center(screen, f"Score: {score}", font, TEXT_COLOR, HEIGHT // 2)
            draw_text_center(screen, f"Coins: {coins_collected}", font, TEXT_COLOR, HEIGHT // 2 + 40)
            draw_text_center(screen, "Pinch to restart", small_font, TEXT_COLOR, HEIGHT // 2 + 90)
            draw_text_center(screen, "L Hand or Rad for car selection", small_font, (200, 200, 255), HEIGHT // 2 + 120)
        pygame.display.flip()
    return False, coins_collected if 'coins_collected' in locals() else 0


def main():
    global total_coins_global, current_skin_index
    pygame.init()
    pygame.display.set_mode((1, 1)) 
    print("=" * 50)
    print("Hand-Controlled Racing Game")
    print("=" * 50)
    print("\nControls:")
    print("1. Show your hand in front of camera")
    print("2. Press 'C' in the camera window to calibrate")
    print("3. Move your hand LEFT/RIGHT to control")
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
        "PlayerRedM.png",
        "Blue Player Car.png",
        "WR Player Car  Scaled.png",
        "Police Player Car.png",
    ]
    for filename in player_car_files:
        img = load_and_scale_image(filename, PLAYER_WIDTH, PLAYER_HEIGHT)
        car_options.append(img)

    flying_animation = load_animation_frames("WR Flying Spritesheet.png", PLAYER_WIDTH, PLAYER_HEIGHT, 69)
    
    if flying_animation is None:
        print("Warning: Could not load flying animation, will use static image")
    if car_options[0] is None:
        print("Warning: Could not load player car, using colored rectangle")
    
    obstacle_imgs = []
    police_car = load_and_scale_image("Enemy Police Race Car Scaled.png", OBSTACLE_WIDTH, OBSTACLE_HEIGHT)
    enemy_blue = load_and_scale_image("EnemyBlue.png", OBSTACLE_WIDTH, OBSTACLE_HEIGHT)
    enemy_green = load_and_scale_image("EnemyGreen.png", OBSTACLE_WIDTH, OBSTACLE_HEIGHT)
    enemy_purple = load_and_scale_image("EnemyPurple.png", OBSTACLE_WIDTH, OBSTACLE_HEIGHT)
    
    if police_car:
        obstacle_imgs.append(police_car)
    else:
        print("Warning: Could not load police car, using colored rectangle")

    if enemy_blue:
        obstacle_imgs.append(enemy_blue)
    else:
        print("Warning: Could not load blue enemy car, using colored rectangle")

    if enemy_purple:
        obstacle_imgs.append(enemy_purple)
    else:
        print("Warning: Could not load purple enemy car, using colored rectangle")

    if enemy_green:
        obstacle_imgs.append(enemy_green)
    else:
        print("Warning: Could not load green enemy car, using colored rectangle")

    if not obstacle_imgs:
        obstacle_imgs = None

    coin_img = load_and_scale_image("Coin2.png", COIN_SIZE, COIN_SIZE)

    if coin_img is None:
        print("Warning: Could not load coin image, using yellow circle")

    powerup_img = load_and_scale_image("powerup.png", POWERUP_SIZE, POWERUP_SIZE)

    if powerup_img is None:
        print("Warning: Could not load powerup image, using magenta circle")

    bg_img = None

    try:
        bg_img = pygame.image.load("Fixed Road 1.png")
        bg_img = pygame.transform.scale(bg_img, (WIDTH, HEIGHT))
    except Exception as e:
        print(f"Error loading background: {e}")

    bg_y_offset = [0]

    if not start_screen(detector):
        detector.stop()
        cap.release()
        cv2.destroyAllWindows()
        pygame.quit()
        return
    
    player_img = car_selection_screen(car_options, flying_animation, detector)
    keep_playing = True

    while keep_playing and detector.running:
        action, coins_earned = game_loop(detector, player_img, flying_animation, obstacle_imgs, bg_img, bg_y_offset, coin_img, powerup_img, car_options)
        total_coins_global += coins_earned
        if action == "restart":
            continue
        elif action == "car_selection":
            player_img = car_selection_screen(car_options, flying_animation, detector)
        else:
            keep_playing = False

    detector.stop()
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()
    print(f"Game closed. Total coins earned: {total_coins_global}")
    print("Thanks for playing!")


if __name__ == "__main__":

    main()
