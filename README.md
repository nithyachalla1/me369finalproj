# ME369 Final Project
This is the I-35 Simulator Game! I-35 Racing Simulator is a hand-gesture–controlled arcade racing game built using Pygame, OpenCV, and MediaPipe Hands.
You control the car by moving your real hand in front of your webcam — no keyboard needed (though keyboard backup controls exist).

Dodge obstacles, collect coins, buy new cars, and grab power-ups to fly over traffic!<br>
<br>


## Features:
- Full hand-tracking controls using MediaPipe
- Lane-switching by hand movement
- Gesture-based menu navigation
- Unlockable cars & in-game shop
- Flying power-up with animated sprite sheet
- Increasing difficulty
- Camera calibration system
- Keyboard fallback controls


<br>

## Required Game Assets

Your working directory must include:

| File                                           | Description               |
| ---------------------------------------------- | ------------------------- |
| `PlayerRedM.png`                               | Default car               |
| `Blue Player Car.png`                          | Unlockable car            |
| `WR Player Car Scaled.png`                     | Unlockable WR car         |
| `Police Player Car.png`                        | Unlockable police car     |
| `WR Flying Spritesheet.png`                    | 69-frame flying animation |
| `EnemyBlue.png`                                | Obstacle                  |
| `EnemyGreen.png`                               | Obstacle                  |
| `EnemyPurple.png`                              | Obstacle                  |
| `Enemy Police Race Car Scaled.png`             | Obstacle                  |
| `coin.png`                                     | Coin sprite               |
| `powerup.png`                                  | Power-up sprite           |
| `Fixed Road 1.png`                             | Scrolling background      |
| `Royalty Free Retro Gaming Music - Racing.mp3` | Music soundtrack          |
| `game_test_2.py`                               | The actual game code      |

Make sure all images are in the same folder as the .py file. <br>
<br>

In order to download all the files from the repo into one folder, click the code dropdown on the home page of the repo, and click "Download ZIP" under local. <br>
<br>
<br>

## How to Run the Game (WINDOWS VERSION)

1) Open your PREFERRED terminal (Conda, Command Prompt, etc.)
2) Navigate to your game directory in terminal:

```
cd path/to/game_folder
```
3) Install dependencies

```
pip install -r requirements.txt
```

4) Run the game:

```
python game_test_2.py
```

5) A camera window will appear. Show your hand clearly 
6) Press C once to calibrate center position
7) The game window opens automatically. Use L-hand gesture to start
<br>
<br>


## How to Run the Game (LINUX VERSION)

1) Open the terminal 
2) Install required packages:

```
sudo apt update
sudo apt install python3 python3-pip python3-venv \
    libgl1 libglib2.0-0 libcanberra-gtk-module libcanberra-gtk3-module
```
3) Create and activate a virtual environment

```
python3 -m venv gameenv
source gameenv/bin/activate
```

4) Install Dependencies:

```
pip install -r requirements.txt
```

5) Enable camera access

Your user should be part of the video group:

```
sudo usermod -a -G video $USER
```

Then log out and back in.

6) Run the game
Inside the game folder

```
python3 game_test_2.py
```

7) Press C once to calibrate center position
8) The game window opens automatically. Use L-hand gesture to start
<br>

## Enjoy!
