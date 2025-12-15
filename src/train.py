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
            time.sleep(0.01)  # Замедление для просмотра
        
        obs = np.concatenate([self.data.qpos, self.data.qvel])
        forward_vel = self.data.qvel[0]
        height = self.data.qpos[2]
        
        reward = forward_vel + (1.0 if 0.7 < height < 1.1 else -1.0)
        terminated = height < 0.5
        
        return obs, reward, terminated, False, {}
    
    def reset(self, seed=None, options=None):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = 0.8
        self.data.qpos[7] = -0.3   # левое колено
        self.data.qpos[10] = -0.3  # правое колено

        # Обнули скорости
        self.data.qvel[:] = 0.0

        mujoco.mj_forward(self.model, self.data)
        
        if self.render_mode and self.viewer:
            self.viewer.sync()
        
        return np.concatenate([self.data.qpos, self.data.qvel]), {}
    
    def close(self):
        if self.viewer:
            self.viewer.close()

# ПРОСТОЙ ВАРИАНТ - используем стандартный model.learn() с callback
if __name__ == "__main__":
    print("="*60)
    print("Обучение с постоянной визуализацией")
    print("="*60)
    
    # Создаем среду
    env = BipedEnv("biped.xml", render=True)
    
    # Создаем модель
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,  # Включаем вывод
        device="cpu"
    )
    
    # Обучаем с визуализацией
    print("\nНачинаю обучение...")
    print("Смотрите окно визуализации")
    print("Нажмите Ctrl+C для остановки\n")
    
    try:
        model.learn(
            total_timesteps=20000,  # Начни с малого
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nОбучение прервано")
    
    # Сохраняем модель
    model.save("biped_trained")
    
    # Тестируем обученную модель
    print("\n" + "="*60)
    print("Тестирование обученной модели")
    print("="*60)
    
    obs, _ = env.reset()
    total_reward = 0
    
    for i in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        
        print(f"\rШаг {i:3d} | Награда: {total_reward:7.2f} | Высота: {env.data.qpos[2]:.3f}", end="")
        
        if terminated:
            print(f"\nУпал на шаге {i}")
            obs, _ = env.reset()
            total_reward = 0
    
    env.close()
    print("\n\nПрограмма завершена!")