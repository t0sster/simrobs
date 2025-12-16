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

        self.prev_y_pos = 0
        
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
        forward_vel = self.data.qvel[1]
        height = self.data.qpos[2]
        y_pos = self.data.qpos[0]
        
        reward = 0.5 * forward_vel + (0.2 if 0.45 < height < 0.6 else -1.0) - 0.5 * (
            np.abs(self.data.qvel[0]) + np.abs(self.data.qvel[2]))
        reward = reward * 0.5 + (y_pos - self.prev_y_pos) * 0.5
        self.prev_y_pos = y_pos
        terminated = height < 0.4 or height > 0.6

        return obs, reward, terminated, False, {}
    
    def reset(self, seed=None, options=None):
        mujoco.mj_resetData(self.model, self.data)

        noise_scale = 0.05

        self.data.qpos[2] = 0.52
        self.data.qpos[9] = -0.6 + np.random.uniform(-noise_scale, noise_scale)
        self.data.qpos[10] = 1.0 + np.random.uniform(-noise_scale, noise_scale)
        self.data.qpos[13] = -0.6 + np.random.uniform(-noise_scale, noise_scale)
        self.data.qpos[14] = 1.0 + np.random.uniform(-noise_scale, noise_scale)

        self.data.qvel[:] = 0.0

        mujoco.mj_forward(self.model, self.data)
        
        if self.render_mode and self.viewer:
            self.viewer.sync()
        
        return np.concatenate([self.data.qpos, self.data.qvel]), {}
    
    def close(self):
        if self.viewer:
            self.viewer.close()

if __name__ == "__main__":
    
    env = BipedEnv("biped.xml", render=True)
    
    try:
        model = PPO.load("biped_trained_100k")
    except:
        env.close()
        exit()
    
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
                # print(env.data.qpos[0:3])
                obs, _ = env.reset()
                total_reward = 0
                
    except KeyboardInterrupt:
        env.close()
    
    env.close()