import time, threading
import multiprocessing

import numpy as np
import pyautogui
import pydirectinput
from PIL import ImageGrab
import cv2

pydirectinput.PAUSE = 0.000001


class Action:
    UP = 0      # hold up and not hold down
    DOWN = 1    # hold down and not hold up
    FORWARD = 2 # not hold up or down
    BOTH = 3    # hold up and hold down


class Environment:
    """
    Environment class is responsible for passing the actions to the game.
    It is also responsible for retrieving the game status and the reward.
    """

    actions = {Action.UP: "f", Action.FORWARD: "", Action.DOWN: "j", Action.BOTH: "fj"}

    def __init__(self, debug=False, deploy=False):
        self.debug = debug
        self.deploy = deploy
        self.queue = multiprocessing.Queue()
        self.client = [
            window
            for window in pyautogui.getWindowsWithTitle("MuseDash")
            if window.title == "MuseDash"
        ][0]
        self.crashed = False
        self.playing = False
        self.combo = False
        self.score = None
        self.last_action = Action.FORWARD
        print("\nGame can be connected (select song)")
        thread_crashed = threading.Thread(target=self.crashed_watcher)
        thread_crashed.daemon = True
        thread_crashed.start()
        thread_score = threading.Thread(target=self.score_watcher)
        thread_score.daemon = True
        thread_score.start()
        thread_combo = threading.Thread(target=self.combo_watcher)
        thread_combo.daemon = True
        thread_combo.start()

    def crashed_watcher(self):
        while True:
            try:
                if self.playing:
                    pyautogui.locateOnScreen(
                        "buttons/minus.png",
                        confidence=0.8,
                        region=(
                            int(self.client.box.left),
                            int(self.client.box.top + self.client.box.height * 0.1),
                            int(self.client.box.height * 0.65),
                            int(self.client.box.width * 0.3),
                        ),
                    )
                    self.crashed = True
                    if self.debug:
                        print("Game crashed")
                else:
                    raise pyautogui.ImageNotFoundException
            except pyautogui.ImageNotFoundException:
                time.sleep(0.05)
                continue

            if not self.deploy:
                self.send_key("esc")
                self.playing = False

    def score_watcher(self):
        while True:
            if self.playing:
                if self.find_score("buttons/perfect.png"):
                    self.score = "perfect"
                elif self.find_score("buttons/great.png"):
                    self.score = "great"
                elif self.find_score("buttons/note.png"):
                    self.score = "perfect"
                elif self.find_score("buttons/pass.png"):
                    self.score = "perfect"
                else:
                    self.score = None
            time.sleep(0.05)

    def combo_watcher(self):
        while True:
            if self.playing:
                if self.find_combo("buttons/combo.png"):
                    self.combo = True
                elif self.find_combo("buttons/combo2.png"):
                    self.combo = True
                else:
                    self.combo = False
            time.sleep(0.05)

    def get_frame(self):
        # Check if window position has changed, update cached region if needed
        if self.client.box.left != getattr(
            self, "_cached_left", None
        ) or self.client.box.top != getattr(self, "_cached_top", None):
            h0, h1 = int(self.client.box.height * 0.22), int(
                self.client.box.height * 0.68
            )
            w0, w1 = int(self.client.box.width * 0.185), int(
                self.client.box.width * 0.54
            )
            left = self.client.box.left + w0
            right = self.client.box.left + w1
            top = self.client.box.top + h0
            bottom = self.client.box.top + h1
            self._screenshot_region = (left, top, right, bottom)
            self._cached_left = self.client.box.left
            self._cached_top = self.client.box.top

        im = ImageGrab.grab(bbox=self._screenshot_region).convert("L")
        return np.asarray(im)

    def get_crashed(self):
        return self.crashed

    def get_score(self):
        return self.score

    def get_combo(self):
        return self.combo

    def send_key(self, key):
        if not self.client.isActive:
            try:
                self.client.activate()
            except:
                pass

        pydirectinput.press(key)

    def click(self, x, y):
        if not self.client.isActive:
            try:
                self.client.activate()
            except:
                pass

        pydirectinput.moveTo(x, y)
        pydirectinput.click()

    def find_image(self, image_path):
        try:
            pyautogui.locateOnScreen(image_path, region=(self.client.box))
            return True
        except:
            return False

    def find_score(self, score_path):
        try:
            pyautogui.locateOnScreen(
                score_path,
                confidence=0.85,
                region=(
                    int(self.client.box.left),
                    int(self.client.box.top + self.client.box.height * 0.1),
                    int(self.client.box.height * 0.65),
                    int(self.client.box.width * 0.3),
                ),
            )
            return True
        except pyautogui.ImageNotFoundException:
            return False

    def find_combo(self, combo_path):
        try:
            pyautogui.locateOnScreen(
                combo_path,
                confidence=0.8,
                grayscale=True,
                region=(
                    int(self.client.box.left + self.client.box.width * 0.4),
                    int(self.client.box.top + self.client.box.height * 0.18),
                    int(self.client.box.width * 0.2),
                    int(self.client.box.height * 0.1),
                ),
            )
            return True
        except pyautogui.ImageNotFoundException:
            return False

    def set_fn_message_received(self):
        image, crashed, score, combo = (
            self.get_frame(),
            self.get_crashed(),
            self.get_score(),
            self.get_combo(),
        )
        # if self.debug: print(f"Incoming data: image, crashed={crashed}, score={score}")

        self.queue.put((image, crashed, score, combo))

    def start_game(self):
        """
        Starts the game and lets the TRex run for half a second and then returns the initial state.

        :return: the initial state of the game (np.array, reward, crashed).
        """
        # game can not be started as long as the browser is not ready
        if self.debug:
            print("Waiting for game to start...")
        self.crashed = False
        self.playing = False
        self.score = None
        while True:
            if self.find_image("buttons/start.png"):
                time.sleep(0.2)
                self.send_key("enter")
                if self.debug:
                    print("Game started")
                break
            if self.find_image("buttons/restart_button.png"):
                time.sleep(0.2)
                self.send_key(["left", "enter"])
                if self.debug:
                    print("Game restarted")
                break
            if self.find_image("buttons/restart.png"):
                time.sleep(0.2)
                self.send_key(["r"])
                if self.debug:
                    print("Game restarted")
                break
            time.sleep(0.2)

        time.sleep(6)
        self.crashed = False
        self.playing = True
        self.score = None
        return self.get_state(Action.FORWARD)

    def do_action(self, action):
        """
        Performs action and returns the updated status
        """        
        if not self.client.isActive:
            try:
                self.client.activate()
            except:
                pass

        for i in self.actions[self.last_action]:
            if i not in self.actions[action]:
                pydirectinput.keyUp(i)
        for i in self.actions[action]:
            if i not in self.actions[self.last_action]:
                pydirectinput.keyDown(i)

        # Capture state after action
        return self.get_state(action)

    def get_state(self, action):
        self.set_fn_message_received()
        image, crashed, score, combo = self.queue.get()
        if self.debug:
            print(
                f"Get state: action={action}, score={score}, combo={combo}, crashed={crashed}"
            )

        if crashed:
            reward = -200.0
        else:
            reward = 3
            if score:
                if score == "perfect":
                    reward = 45
                elif score == "great":
                    reward = 15

            # Reward for no unnecessary actions
            if action == Action.FORWARD:
                reward = 10

            # Penalty for violent double-tapping scores
            if (
                action == Action.BOTH
                or (action == Action.DOWN and self.last_action == Action.UP)
                or (action == Action.UP and self.last_action == Action.DOWN)
            ):
                reward = reward * 0.5

            # Reward for not missing notes
            if combo:
                reward = reward * 1.5

        self.last_action = action

        return image, reward, crashed



class Preprocessor:

    def __init__(self, width, height):
        self.width = width
        self.height = height

    def process(self, frame):
        # Use faster interpolation method
        processed = cv2.resize(
            frame, (self.width, self.height), interpolation=cv2.INTER_NEAREST
        )

        # Randomly adjust brightness and contrast
        alpha = np.random.uniform(0.8, 1.2)
        processed = cv2.convertScaleAbs(processed, alpha=alpha)

        # Randomly invert the image
        if np.random.random() < 0.5:
            processed = np.invert(processed)

        # Random Gaussian blur
        if np.random.random() < 0.5:
            kernel_size = np.random.choice([3, 5, 7])
            processed = cv2.GaussianBlur(processed, (kernel_size, kernel_size), 0)

        processed = processed / 255.0
        return processed.astype(np.float32)

    def get_initial_state(self, first_frame):
        self.state = np.array([first_frame, first_frame, first_frame, first_frame])
        return self.state

    def get_updated_state(self, next_frame):
        self.state = np.array([*self.state[-3:], next_frame])
        return self.state
