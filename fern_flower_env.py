import easyocr
import pyautogui
import random
import time
import threading
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from PIL import Image
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env


class FernFlowerEnv(gym.Env):
    def __init__(self, close_image_path: str, play_image_path: str, game_speed: int = 1):
        self.reader = easyocr.Reader(['en'])
        self.dest_ss_size = (720, 1280)
        self.close_image_pattern_np = np.array(Image.open(close_image_path)).astype('uint8')
        self.close_button_center = (596, 1117)
        self.play_image_pattern_np = np.array(Image.open(play_image_path)).astype('uint8')
        self.play_button_center = (360, 973)
        self.d = 80
        self.accepted_fails = 50
        
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

        self.curr_result_img = np.zeros((45, 120))
        self.result_changes = 0
        self.screenshots_wo_change = 0
        self.stuck_game_threshold = 50

        self.ai_ss_shape = (1, 160, 90)
        self.ai_ss_size = (90, 160)
        self.last_ai_ss = np.zeros(self.ai_ss_shape)
        self.action_space = spaces.Discrete(len(self.directions) * len(self.jump_time_dict))
        self.observation_space = spaces.Box(low=0, high=255, shape=self.ai_ss_shape, dtype=np.uint8)
        self.done = False

    def take_screenshot(self):
        screenshot = pyautogui.screenshot()
        width, height = screenshot.size
        screenshot = screenshot.crop((1, 1, width-1, height-1)).convert("L") 
        screenshot_bw_np = np.array(screenshot)
        non_zero_columns = np.any(screenshot_bw_np != 0, axis=0)
        screen_only_np = screenshot_bw_np[:, non_zero_columns]
        screen_only = Image.fromarray(screen_only_np).resize(self.dest_ss_size)
        return screen_only

    @staticmethod
    def extract_whites(image):
        image_np_mask = np.where(np.array(image) == 255, 255, 0).astype('uint8')
        return image_np_mask
        
    def extract_results_from_ss(self, screenshot):
        x_start, x_end, y_start, y_end = 240, 460, 550, 620
        result_img = screenshot.crop((x_start, y_start, x_end, y_end))
        result = self.reader.readtext(np.array(result_img))
        numbers = ''.join([text[1] for text in result if text[1].isdigit()])
    
        try:
            numbers = int(numbers)
            return numbers
        except:
            return -1

    def compare_image_with_pattern(self, cropped_image_np, pattern_np):
        screenshot_wo_np = self.extract_whites(cropped_image_np)
        bool_mask = screenshot_wo_np == pattern_np
        if np.sum(bool_mask == False) < self.accepted_fails:
            return True
        return False

    def is_result_changed(self, screenshot: Image):
        x_start, x_end, y_start, y_end = 530, 650, 45, 90
        tmp_result = screenshot.crop((x_start, y_start, x_end, y_end))
        tmp_result_np = self.extract_whites(tmp_result)
        if not self.compare_image_with_pattern(tmp_result_np, self.curr_result_img):
            self.curr_result_img = tmp_result_np
            self.result_changes += 1
            self.screenshots_wo_change = 0
            return True

        else:
            self.screenshots_wo_change += 1
            return False

    def is_revive_screen(self, screenshot: Image, possible_fails:int = 50):
        close_x, close_y = self.close_button_center
        x_start, x_end, y_start, y_end = close_x - self.d, close_x + self.d, close_y - self.d, close_y + self.d
        cropped_image_np = screenshot.crop((x_start, y_start, x_end, y_end))
        return self.compare_image_with_pattern(cropped_image_np, self.close_image_pattern_np)

    def is_endgame_screen(self, screenshot: Image, possible_fails:int = 50):
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
            self.curr_result_img = np.zeros((45, 120))

    def jump_in_background(self, direction, time_code):
        thread = threading.Thread(target=self.jump, args=(direction, time_code))
        thread.start()
        return thread 

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

    def check_game_end_in_background(self):
        result = [None]
        def worker():
            result[0] = self.check_game_end()
        thread = threading.Thread(target=worker)
        thread.start()
        return thread, result

    def run_sample(self):
        self.click(self.close_button_position)
        time.sleep(self.reload_gui_time_buffer)
        self.click(self.play_button_position)
        time.sleep(self.reload_gui_time_buffer)

        games_completed = 0
        while games_completed < 20:
            jump_thread = self.jump_in_background(random.choice(self.directions), random.choice(list(self.jump_time_dict.keys())))
            check_thread, result_container = self.check_game_end_in_background()
            jump_thread.join()
            check_thread.join()
            is_game_end, result_changed, result = result_container[0]
          
            if self.screenshots_wo_change >= self.stuck_game_threshold:
                time.sleep(2 * self.reload_gui_time_buffer) # let it die
                print("STUCKED GAME")
            
            if is_game_end:
                print(f"END GAME: [{result} pts] [{self.result_changes} result changes]")
                self.result_changes = 0
                games_completed += 1          
    

    def step(self, action):
        jump_time_code, jump_direction = self.actions[action]
        jump_thread = self.jump_in_background(jump_direction, jump_time_code)
        check_thread, result_container = self.check_game_end_in_background()
        jump_thread.join()
        check_thread.join()
        is_game_end, result_changed, result = result_container[0]
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
        self.curr_result_img = np.zeros((45, 120))
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
    models_dir = "models/PPO"
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

    env.close()
    mean_reward = cumulated_rewards/episode
    mean_result = cumulated_results/episode
    return mean_reward, mean_result


def test_and_evaluate(env, episodes=100, filename='evalutaion.txt'):
    for iter_log in range(55000, 61000, 1000):
        model_path = f"models/PPO/{str(iter_log)}"
        mean_reward, mean_result = evaluate_model(env, model_path, episodes)
        result_arr = [str(txt) for txt in [episodes, model_path, mean_reward, mean_result]]
        result_str = ','.join(result_arr) + '\n'
        with open(filename, "a") as file:
            file.write(result_str)


if __name__ == "__main__":
    env = FernFlowerEnv("base_images/close_button_bin.png", "base_images/play_button_bin.png", 2)
    # check_env(env)
    model_path = "models/PPO/55000"
    # mean_reward, mean_result = evaluate_model(env, model_path, 5)
    # print(f"Mean reward: {mean_reward} Mean result: {mean_result}")
    test_and_evaluate(env, episodes=10, filename="test_eval_55_60.txt")
    # train_env(env, model_path, 47, 100)
