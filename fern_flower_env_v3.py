import easyocr
import pyautogui
import random
import time
import threading
import os
import numpy as np
import gymnasium as gym
import mss
from gymnasium import spaces
from PIL import Image
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env


class FernFlowerEnv3(gym.Env):
    def __init__(self, close_image_path: str, play_image_path: str, game_speed: int = 1):
        self.reader = easyocr.Reader(['en'])
        self.dest_ss_size = (360, 640)
        self.close_image_pattern_np = np.array(Image.open(close_image_path)).astype('uint8')
        self.close_button_center = (298, 558)
        self.play_image_pattern_np = np.array(Image.open(play_image_path)).astype('uint8')
        self.play_button_center = (180, 486)
        self.d = 40
        self.accepted_fails = 80
        
        screen_width, screen_height = pyautogui.size()
        self.play_button_position = (int(0.5 * screen_width), int(0.75 * screen_height))
        self.close_button_position = (int(0.6 * screen_width), int(0.87 * screen_height))

        self.game_speed = game_speed

        self.jump_time_dict = {
            'short': 0.05/game_speed,
            'medium': 0.15/game_speed,
            'long': 0.3/game_speed
        }
        self.reload_gui_time_buffer = 1/game_speed
        self.directions = ['left', 'right']

        self.actions = {
            0: ('no_jump', ''),
            1: ('short', 'left'),
            2: ('medium', 'left'),
            3: ('long', 'left'),
            4: ('short', 'right'),
            5: ('medium', 'right'),
            6: ('long', 'right'),
        }

        self.curr_result_img = np.zeros((24, 60))
        self.result_changes = 0
        self.screenshots_wo_change = 0
        self.stuck_game_threshold = 50

        self.ai_ss_shape = (1, 160, 90)
        self.ai_ss_size = (90, 160)
        self.last_ai_ss = np.zeros(self.ai_ss_shape)
        self.action_space = spaces.Discrete(len(self.directions) * len(self.jump_time_dict))
        self.observation_space = spaces.Box(low=0, high=255, shape=self.ai_ss_shape, dtype=np.uint8)
        self.done = False

        self.region = self._precalc_region()
        self.sct = mss.mss()


    def _precalc_region(self):
        width, height = pyautogui.size()        
        dist_from_center = 9 * height / 32 # for 16:9 display proportions
        screen_center = width/2
        start_width = int(screen_center - dist_from_center)
        return {"top": 1, "left": start_width, "width": int(dist_from_center*2), "height": int(height-2)} 
        
    def take_screenshot(self):
        screenshot = self.sct.grab(self.region)
        screenshot_np = np.array(screenshot)
        grayscale_screenshot = Image.fromarray(screenshot_np).convert("L")  # Convert to grayscale (L mode)
        resized_screenshot = grayscale_screenshot.resize(self.dest_ss_size)
        return resized_screenshot

    @staticmethod
    def extract_whites(image):
        image_np_mask = np.where(np.array(image) > 200, 255, 0).astype('uint8')
        return image_np_mask
        
    def extract_results_from_ss(self, screenshot):
        x_start, x_end, y_start, y_end = 120, 230, 275, 310
        result_img = screenshot.crop((x_start, y_start, x_end, y_end))
        result = self.reader.readtext(np.array(result_img))
        numbers = ''.join([text[1] for text in result if text[1].isdigit()])
    
        try:
            numbers = int(numbers)
            return numbers
        except:
            return -1

    def compare_image_with_pattern(self, cropped_image_np, pattern_np, accepted_fails = None):
        if not accepted_fails:
            accepted_fails = self.accepted_fails

        screenshot_wo_np = self.extract_whites(cropped_image_np)
        bool_mask = screenshot_wo_np == pattern_np
        if np.sum(bool_mask == False) < accepted_fails:
            return True
        return False

    def is_result_changed(self, screenshot: Image):
        x_start, x_end, y_start, y_end = 265, 325, 22, 46
        tmp_result = screenshot.crop((x_start, y_start, x_end, y_end))               
        if not self.compare_image_with_pattern(tmp_result,  self.curr_result_img, 20):
            self.curr_result_img = self.extract_whites(tmp_result)
            self.result_changes += 1
            self.screenshots_wo_change = 0
            return True

        else:
            self.screenshots_wo_change += 1
            return False

    def is_revive_screen(self, screenshot: Image):
        close_x, close_y = self.close_button_center
        x_start, x_end, y_start, y_end = close_x - self.d, close_x + self.d, close_y - self.d, close_y + self.d
        cropped_image_np = screenshot.crop((x_start, y_start, x_end, y_end))
        return self.compare_image_with_pattern(cropped_image_np, self.close_image_pattern_np)

    def is_endgame_screen(self, screenshot: Image):
        play_x, play_y = self.play_button_center     
        x_start, x_end, y_start, y_end = play_x - self.d, play_x + self.d, play_y - self.d, play_y + self.d 
        cropped_image_np = screenshot.crop((x_start, y_start, x_end, y_end))
        return self.compare_image_with_pattern(cropped_image_np, self.play_image_pattern_np)

    @staticmethod
    def click(coordinates):
        pyautogui.click(*coordinates)
    
    def jump(self, direction, time_code):
        if not time_code == 'no_jump':
            jump_time_float = self.jump_time_dict[time_code]
            pyautogui.keyDown(direction)
            time.sleep(jump_time_float)
            pyautogui.keyUp(direction)

        if self.screenshots_wo_change >= self.stuck_game_threshold:
            time.sleep(2 * self.reload_gui_time_buffer) # let it die
            self.screenshots_wo_change = 0 # to avoid stucking on ground landing on lower results (15 or below)
            self.curr_result_img = np.zeros((24, 60))

    def check_game_end(self):
        screenshot = self.take_screenshot()
        result_changed = self.is_result_changed(screenshot)
        self.last_ai_ss = np.expand_dims(np.array(screenshot.resize(self.ai_ss_size)).astype('uint8'), axis=0)
       
        if self.is_revive_screen(screenshot):
            self.click(self.close_button_position)
            time.sleep(self.reload_gui_time_buffer)
                
        if self.is_endgame_screen(screenshot):
            time.sleep(self.reload_gui_time_buffer)
            results_screenshot = self.take_screenshot()
            result = self.extract_results_from_ss(results_screenshot)
            self.click(self.play_button_position)
            time.sleep(self.reload_gui_time_buffer)
            return True, True, result
        
        return False, result_changed, 0

    def step(self, action):
        jump_time_code, jump_direction = self.actions[action]
        self.jump(jump_direction, jump_time_code)
        is_game_end, result_changed, result = self.check_game_end()
        observation = self.last_ai_ss
        reward = 1 if result_changed else 0
        reward += 0 if result <=100 else int(((result-100)**2)/10000)
        self.done = is_game_end
        terminated = is_game_end
        truncated = False
        return observation, reward, terminated, truncated, {'result': result}
    
    def reset(self, seed=42):
        self.done = False
        self.result_changes = 0
        self.screenshots_wo_change = 0
        self.curr_result_img = np.zeros((24, 60))
        initial_observation = np.expand_dims(np.array(self.take_screenshot().resize(self.ai_ss_size)).astype('uint8'), axis=0)
        self.click(self.play_button_position)
        return initial_observation, {}
    
    def render(self, mode='human'):
        if mode == 'human':
            img = Image.fromarray(self.last_ai_ss)
            img.show()
    
    def close(self):
        pass


def train_env(env, model_path=None, model_first_step=0, model_last_step=30): 
    env.reset()
    models_dir = "models/PPOv3"
    logdir = "logs"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    if not model_path:
        model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=logdir)
        model_first_step = 0
    else: 
        model = PPO.load(model_path, env=env)

    TIMESTEPS = 1000
    for i in range(model_first_step, model_last_step):
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
        model.save(f"{models_dir}/{TIMESTEPS*i}")


def evaluate_model(env, model_path, episodes=10):
    model = PPO.load(model_path, env)
    env.reset()
    cumulated_results = 0
    cumulated_rewards = 0
    max_result = 0

    for episode in range(episodes):
        obs, _dct = env.reset()
        terminated = False
        total_reward = 0

        while not terminated:
            action, _states = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            
        episode_result = info['result']
        print(f"Episode: {episode + 1} Total reward: {total_reward} Result: {episode_result}")
        cumulated_rewards += total_reward
        cumulated_results += episode_result
        max_result = max(max_result, episode_result)

    env.close()
    mean_reward = cumulated_rewards/episode
    mean_result = cumulated_results/episode
    return mean_reward, mean_result, max_result


def test_and_evaluate(env, start_range, end_range, episodes=100, filename='evalutaion.txt'):
    for iter_log in range(start_range, end_range, 1000):
        model_path = f"models/PPOv3/{str(iter_log)}"
        mean_reward, mean_result, max_result = evaluate_model(env, model_path, episodes)
        result_arr = [str(txt) for txt in [episodes, model_path, mean_reward, mean_result, max_result]]
        result_str = ','.join(result_arr) + '\n'
        with open(filename, "a") as file:
            file.write(result_str)


if __name__ == "__main__":
    env = FernFlowerEnv3("base_images/close_button_bin_v3.png", "base_images/play_button_bin_v3.png", 2)
    check_env(env)
    # train_env(env, None, 0, 60)
    test_and_evaluate(env, 49000, 52000, episodes=50, filename="evalutaion_v3.txt")
