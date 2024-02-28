import carla
import gym
import time
import random
import numpy as np
import math
from queue import Queue
from gym import spaces
from absl import logging
import pygame


from carla_utils.graphics import HUD
from carla_utils.utils import get_actor_display_name, smooth_action
from core_rl.actions import CarlaActions
from core_rl.observation import CarlaObservations

logging.set_verbosity(logging.INFO)

# Carla environment
class CarlaEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, town, fps, obs_width, obs_height, view_height, view_width, repeat_action, sensors,
                 action_type, enable_preview, steps_per_episode, playing=False, allow_render=True, allow_spectator=True):

        self.obs_height = obs_height
        self.obs_width = obs_width
        self.spectator_height = view_height
        self.spectator_width = view_width
        self.allow_render = allow_render
        self.allow_spectator = allow_spectator
        self.spectator_camera = None
        self.tesla = blueprint_library.filter('model3')[0]
        self.episode_idx = -2
        self.world = None
        self.actions = CarlaActions()
        self.observations = CarlaObservations(self.obs_height, self.obs_width)

        self.action_space = self.actions.get_action_space()
        self.observation_space = self.observations.get_observation_space()

        try:
            self.client = carla.Client("192.168.0.10", 2000)  
            self.client.set_timeout(100.0)

            self.client.load_world(map_name=town)
            self.world = self.client.get_world()
            self.world.set_weather(carla.WeatherParameters.ClearNoon)  
            self.world.apply_settings(
                carla.WorldSettings(  
                    synchronous_mode=True,
                    fixed_delta_seconds=1.0 / fps,
                ))
            self.client.reload_world(False)  # reload map keeping the world settings

            # Spawn Vehicle
            self.start_transform = self._get_start_transform()
            self.curr_loc = self.start_transform.location
            self.vehicle = self.world.spawn_actor(self.tesla, self.start_transform)

           # Spawn collision and invasion sensors
            colsensor = self.world.get_blueprint_library().find('sensor.other.collision')
            lanesensor = self.world.get_blueprint_library().find('sensor.other.lane_invasion')
            self.colsensor = self.world.spawn_actor(colsensor, carla.Transform(), attach_to=self.vehicle)
            self.lanesensor = self.world.spawn_actor(lanesensor, carla.Transform(), attach_to=self.vehicle)
            self.colsensor.listen(self._collision_data)
            self.lanesensor.listen(self._lane_invasion_data)  

            # Create hud and initialize pygame for visualization
            if self.allow_render:
                pygame.init()
                pygame.font.init()
                self.display = pygame.display.set_mode((self.spectator_width, self.spectator_height), pygame.HWSURFACE | pygame.DOUBLEBUF)
                self.clock = pygame.time.Clock()
                self.hud = HUD(self.spectator_width, self.spectator_height)
                self.hud.set_vehicle(self.vehicle)
                self.world.on_tick(self.hud.on_world_tick)

            # Set observation image
                      
 
            # Set spectator cam   
            if self.allow_spectator:
                self.spectator_camera = self.world.get_blueprint_library().find('sensor.camera.rgb')
                self.spectator_camera.set_attribute('image_size_x', '800')
                self.spectator_camera.set_attribute('image_size_y', '800')
                self.spectator_camera.set_attribute('fov', '100')
                transform = carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0))
                self.spectator_sensor = self.world.spawn_actor(self.spectator_camera, transform, attach_to=self.vehicle, attachment_type=carla.AttachmentType.SpringArm)
                self.spectator_sensor.listen(self.spectator_image_Queue.put)
                    
        except RuntimeError as msg:
            pass

        #self.server = self.get_pid("CarlaUE4-Linux-Shipping")
        self.map = self.world.get_map()
        blueprint_library = self.world.get_blueprint_library()
        
        self.obs_width = obs_width
        self.obs_height = obs_height
        self.repeat_action = repeat_action
        self.sensors = sensors
        self.actor_list = []
        
        self.steps_per_episode = steps_per_episode
        self.playing = playing
        self.preview_camera_enabled = enable_preview


    # Resets environment for new episode
    def reset(self):
        # logging.debug("Resetting environment")
        # Car, sensors, etc. We create them every episode then destroy
        self.episode_idx += 1
        self.terminate = False
        self.extra_info = []  # List of extra info shown on the HUD
        self.observation = self.observation_buffer = None  # Last received observation
        self.viewer_image = self.viewer_image_buffer = None  # Last received image to show in the viewer
        self.frame_step = 0
        self.out_of_loop = 0
        self.dist_from_start = 0

        # self.episode += 1

        # Append actor to a list of spawned actors, we need to remove them later
        self.actor_list.append(self.vehicle)

        # TODO: combine the sensors
        if 'rgb' in self.sensors:
            self.rgb_cam = self.world.get_blueprint_library().find('sensor.camera.rgb')
        elif 'semantic' in self.sensors:
            self.rgb_cam = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        else:
            raise NotImplementedError('unknown sensor type')

        self.rgb_cam.set_attribute('image_size_x', f'{self.obs_width}')
        self.rgb_cam.set_attribute('image_size_y', f'{self.obs_height}')
        self.rgb_cam.set_attribute('fov', '90')

        bound_x = self.vehicle.bounding_box.extent.x
        bound_y = self.vehicle.bounding_box.extent.y


        transform_front = carla.Transform(carla.Location(x=bound_x, z=1.0))
        self.sensor_front = self.world.spawn_actor(self.rgb_cam, transform_front, attach_to=self.vehicle)
        self.sensor_front.listen(self.front_image_Queue.put)
        self.actor_list.extend([self.sensor_front])

        # Preview ("above the car") camera
        if self.preview_camera_enabled:
            # TODO: add the configs
            
            self.preview_sensor.listen(self.preview_image_Queue.put)
            self.actor_list.append(self.preview_sensor)

        # Disengage brakes
        self.vehicle.apply_control(carla.VehicleControl(brake=0.0))

        image = self.front_image_Queue.get()
        image = np.array(image.raw_data)
        image = image.reshape((self.obs_height, self.obs_width, -1))
        image = image[:, :, :3]

        self.world.tick()
        # Return initial observation
        time.sleep(0.2)
        obs = self.step(None)[0]
        time.sleep(0.2)
        return obs

        return image

    def step(self, action):
        total_reward = 0
        obs, rew, done, info = self._step(action)
        total_reward += rew

        return obs, total_reward, done, info

    # Steps environment
    def _step(self, action):
        self.world.tick()
        self.render()
            
        self.frame_step += 1
        print(action)
        # Apply control to the vehicle based on an action
        control = carla.VehicleControl()
        control.throttle = min(1.0, max(0.0, action[0].item()))
        control.brake = min(1.0, max(0.0, action[1].item()))
        control.steer = min(1.0, max(-1, action[2].item()))
        self.vehicle.apply_control(control)

        # Calculate speed in km/h from car's velocity (3D vector)
        v = self.vehicle.get_velocity()
        kmh = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)

        loc = self.vehicle.get_location()
        new_dist_from_start = loc.distance(self.start_transform.location)
        square_dist_diff = new_dist_from_start ** 2 - self.dist_from_start ** 2
        self.dist_from_start = new_dist_from_start

        image = self.front_image_Queue.get()
        image = np.array(image.raw_data)
        image = image.reshape((self.obs_height, self.obs_width, -1))

        # TODO: Combine the sensors
        if 'rgb' in self.sensors:
            image = image[:, :, :3]
        if 'semantic' in self.sensors:
            image = image[:, :, 2]
            image = (np.arange(13) == image[..., None])
            image = np.concatenate((image[:, :, 2:3], image[:, :, 6:8]), axis=2)
            image = image * 255

        done = False
        reward = 0
        info = dict()

        # # If car collided - end and episode and send back a penalty
        if len(self.collision_hist) != 0:
            done = True
            reward += -100
            self.collision_hist = []
            self.lane_invasion_hist = []

        if len(self.lane_invasion_hist) != 0:
            reward += -5
            self.lane_invasion_hist = []

        # # Reward for speed
        # if not self.playing:
        #     reward += 0.1 * kmh * (self.frame_step + 1)
        # else:
        #     reward += 0.1 * kmh

        reward += 0.1 * kmh

        reward += square_dist_diff

        # # Reward for distance to road lines
        # if not self.playing:
        #     reward -= math.exp(-dis_to_left)
        #     reward -= math.exp(-dis_to_right)
        
        if self.frame_step >= self.steps_per_episode:
            done = True

        #if not self._on_highway():
        #    self.out_of_loop += 1
        #    if self.out_of_loop >= 20:
        #        done = True
        #else:
        #    self.out_of_loop = 0

        # self.total_reward += reward

        if done:
            # info['episode'] = {}
            # info['episode']['l'] = self.frame_step
            # info['episode']['r'] = reward
            logging.warn("Env lasts {} steps, restarting ... ".format(self.frame_step))
            self._destroy_agents()
        
        return image, reward, done, info
    
    def close(self):
        '''
        logging.info("Closes the CARLA server with process PID {}".format(self.server))
        os.killpg(os.getpgid(self.server), signal.SIGKILL)
        atexit.unregister(lambda: os.killpg(os.getpgid(self.server), signal.SIGKILL))
        '''
        pass
    
    def render(self, mode="human"):
 
       # Tick render clock
       self.clock.tick()
       self.hud.tick(self.world, self.clock)
  
       # Add metrics to HUD
       self.extra_info.extend([
           "Episode {}".format(self.episode_idx),
           "Reward: % 19.2f" % self.last_reward,
           "",
           "Maneuver:        % 11s" % maneuver,
           "Routes completed:    % 7.2f" % self.routes_completed,
           "Distance traveled: % 7d m" % self.distance_traveled,
           "Center deviance:   % 7.2f m" % self.distance_from_center,
           "Avg center dev:    % 7.2f m" % (self.center_lane_deviation / self.step_count),
           "Avg speed:      % 7.2f km/h" % (self.speed_accum / self.step_count),
           "Total reward:        % 7.2f" % self.total_reward,
       ])
       if self.allow_spectator:
           # Blit image from spectator camera
           self.viewer_image = self._draw_path(self.camera, self.viewer_image)
           self.display.blit(pygame.surfarray.make_surface(self.viewer_image.swapaxes(0, 1)), (0, 0))
           # Superimpose current observation into top-right corner
       obs_h, obs_w = self.observation.shape[:2]
       pos_observation = (self.display.get_size()[0] - obs_w - 10, 10)
       self.display.blit(pygame.surfarray.make_surface(self.observation.swapaxes(0, 1)), pos_observation)
       pos_vae_decoded = (self.display.get_size()[0] - 2 * obs_w - 10, 10)
       if self.decode_vae_fn:
           self.display.blit(pygame.surfarray.make_surface(self.observation_decoded.swapaxes(0, 1)), pos_vae_decoded)
       if self.activate_lidar:
           lidar_h, lidar_w = self.lidar_data.shape[:2]
           pos_lidar = (self.display.get_size()[0] - obs_w - 10, 100)
           self.display.blit(pygame.surfarray.make_surface(self.lidar_data.swapaxes(0, 1)), pos_lidar)
       # Render HUD
       self.hud.render(self.display, extra_info=self.extra_info)
       self.extra_info = []  # Reset extra info list
       # Render to screen
       pygame.display.flip()

    def _destroy_agents(self):

        for actor in self.actor_list:

            # If it has a callback attached, remove it first
            if hasattr(actor, 'is_listening') and actor.is_listening:
                actor.stop()

            # If it's still alive - desstroy it
            if actor.is_alive:
                actor.destroy()

        self.actor_list = []

    def _collision_data(self, event):

        # What we collided with and what was the impulse
        if get_actor_display_name(event.other_actor) != "Road":
            self.terminate = True
        if self.allow_render:
            self.hud.notification("Collision with {}".format(get_actor_display_name(event.other_actor)))

        #collision_impulse = math.sqrt(event.normal_impulse.x ** 2 + event.normal_impulse.y ** 2 + event.normal_impulse.z ** 2)

        
    def _lane_invasion_data(self, event):

        self.terminate = True
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ["%r" % str(x).split()[-1] for x in lane_types]
        if self.allow_render:
            self.hud.notification("Crossed line %s" % " and ".join(text))

    def _get_observation(self):
        while self.observation_buffer is None:
            pass
        obs = self.observation_buffer.copy()
        self.observation_buffer = None
        return obs

    def _get_viewer_image(self):
        while self.viewer_image_buffer is None:
            pass
        image = self.viewer_image_buffer.copy()
        self.viewer_image_buffer = None
        return image

    def _get_start_transform(self):
        return random.choice(self.map.get_spawn_points())    
            
    
        