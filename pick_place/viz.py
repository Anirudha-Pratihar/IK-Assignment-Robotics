"""Offscreen rendering helpers: screenshots (failure cases) and mp4 videos.  Needs a GL backend:
   Linux headless:  export MUJOCO_GL=egl   (or osmesa)            desktop: nothing to do."""
import os
import numpy as np
import mujoco


class Recorder:
    def __init__(self, env, width=960, height=540, fps=30, azimuth=150, elevation=-25, distance=1.15, lookat=None):
        self.env, self.fps = env, fps
        self.renderer = mujoco.Renderer(env.model, height, width)
        self.cam = mujoco.MjvCamera(); self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        d = env.design
        self.cam.lookat[:] = lookat if lookat is not None else [d["cx"] * 0.8, d["cy"], d["H"] + 0.12]
        self.cam.distance, self.cam.azimuth, self.cam.elevation = distance, azimuth, elevation
        self.frames, self.next_t, self.label = [], 0.0, ""
        self.opt = mujoco.MjvOption()

    def frame(self, label=None):
        self.renderer.update_scene(self.env.data, camera=self.cam, scene_option=self.opt)
        img = self.renderer.render().copy()
        txt = label if label is not None else self.label
        if txt:
            try:
                import cv2
                cv2.putText(img, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            except Exception: pass
        return img

    def maybe_capture(self, env):
        if env.data.time >= self.next_t:
            self.frames.append(self.frame()); self.next_t += 1.0 / self.fps

    def save_png(self, path, label=None):
        import imageio.v2 as imageio
        os.makedirs(os.path.dirname(path), exist_ok=True); imageio.imwrite(path, self.frame(label))

    def save_video(self, path):
        import imageio.v2 as imageio
        os.makedirs(os.path.dirname(path), exist_ok=True)
        imageio.mimsave(path, self.frames, fps=self.fps, macro_block_size=1)
        n = len(self.frames); self.frames = []; return n
