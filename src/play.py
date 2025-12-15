import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import time

class BipedEnv(gym.Env):
    def __init__(self, xml_path="biped.xml", render=True):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.model.opt.timestep = 0.002
        self.data = mujoco.MjData(self.model)
        
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, 
            shape=(self.model.nu,),
            dtype=np.float32
        )
        
        obs_dim = self.model.nq + self.model.nv
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(obs_dim,), 
            dtype=np.float32
        )
        
        self.render_mode = render
        self.viewer = None
        if self.render_mode:
            self._init_viewer()
    
    def _init_viewer(self):
        import mujoco.viewer
        print("Инициализация окна визуализации...")
        self.viewer = mujoco.viewer.launch_passive(
            self.model, 
            self.data,
            show_left_ui=False,
            show_right_ui=False
        )
        time.sleep(0.5)
    
    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.data.ctrl[:] = action * 30
        
        mujoco.mj_step(self.model, self.data)
        
        if self.render_mode and self.viewer:
            self.viewer.sync()
            time.sleep(0.01)
        
        obs = np.concatenate([self.data.qpos, self.data.qvel])
        forward_vel = self.data.qvel[0]
        height = self.data.qpos[2]
        
        reward = forward_vel + (1.0 if height > 0.7 else 0.0)
        terminated = height < 0.6
        
        return obs, reward, terminated, False, {}
    
    def reset(self, seed=None, options=None):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = 0.8
        self.data.qpos[7] = 0.0   # левое колено
        self.data.qpos[10] = 0.0  # правое колено

        # Обнули скорости
        self.data.qvel[:] = 0.0

        mujoco.mj_forward(self.model, self.data)
        
        if self.render_mode and self.viewer:
            self.viewer.sync()
        
        return np.concatenate([self.data.qpos, self.data.qvel]), {}
    
    def close(self):
        if self.viewer:
            self.viewer.close()

# ТОЛЬКО ВИЗУАЛИЗАЦИЯ ОБУЧЕННОЙ МОДЕЛИ
if __name__ == "__main__":
    print("="*60)
    print("Визуализация обученной модели")
    print("="*60)
    
    # Создаем среду
    env = BipedEnv("biped.xml", render=True)
    
    # Загружаем обученную модель
    try:
        model = PPO.load("biped_trained")
        print("Модель 'biped_trained.zip' загружена")
    except:
        print("Ошибка: файл 'biped_trained.zip' не найден!")
        print("Сначала обучите модель с помощью main_script.py")
        env.close()
        exit()
    
    # Визуализация
    print("\nНачинаю визуализацию...")
    print("Закройте окно MuJoCo для остановки")
    print("\nУправление:")
    print("- Нажмите Ctrl+C в терминале для выхода")
    print("- Или закройте окно MuJoCo")
    print("="*60)
    
    obs, _ = env.reset()
    total_reward = 0
    episode = 0
    
    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            
            if terminated:
                episode += 1
                print(f"Эпизод {episode}: награда = {total_reward:.1f}")
                obs, _ = env.reset()
                total_reward = 0
                
    except KeyboardInterrupt:
        print("\nВизуализация остановлена пользователем")
    
    env.close()
    print("Программа завершена")