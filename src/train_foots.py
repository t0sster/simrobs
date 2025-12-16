import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import time

class BipedEnv(gym.Env):
    def __init__(self, xml_path="biped_foots.xml", render=True):
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
        action = np.clip(action, -1, 1)
        self.data.ctrl[:] = action * 50
        
        mujoco.mj_step(self.model, self.data)
        
        if self.render_mode and self.viewer:
            self.viewer.sync()
            time.sleep(0.01)
        
        obs = np.concatenate([self.data.qpos, self.data.qvel])
        forward_vel = self.data.qvel[1]
        height = self.data.qpos[2]
        y_pos = self.data.qpos[1]
        
        reward = forward_vel + (0.2 if 0.45 < height < 0.6 else -1.0) - 0.25 * (
            np.abs(self.data.qvel[0]) + np.abs(self.data.qvel[2]))
        reward = reward * 0.5 + (y_pos - self.prev_y_pos) * 0.5
        self.prev_y_pos = y_pos
        terminated = height < 0.4 or height > 0.6
        # print(self.data.qvel, '\n')
        return obs, reward, terminated, False, {}
    
    def reset(self, seed=None, options=None):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = 0.52
        self.data.qpos[9] = -0.6
        self.data.qpos[10] = 1.0
        self.data.qpos[14] = -0.6
        self.data.qpos[15] = 1.0

        self.data.qvel[:] = 0.0

        mujoco.mj_forward(self.model, self.data)
        
        if self.render_mode and self.viewer:
            self.viewer.sync()
        
        return np.concatenate([self.data.qpos, self.data.qvel]), {}
    
    def close(self):
        if self.viewer:
            self.viewer.close()


if __name__ == "__main__":
    
    env = BipedEnv("biped_foots.xml", render=True)
    
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
        verbose=1,
        device="cpu"
    )
        
    try:
        model.learn(
            total_timesteps=10000,
            progress_bar=True
        )
    except KeyboardInterrupt:
        env.close()
    
    model.save("biped_foots_trained")
        
    obs, _ = env.reset()
    total_reward = 0
    
    for _ in range(25):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        
        if terminated:
            obs, _ = env.reset()
            total_reward = 0
    
    env.close()