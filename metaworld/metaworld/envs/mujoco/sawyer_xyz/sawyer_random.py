from collections import OrderedDict
import numpy as np
from gym.spaces import  Dict , Box

from metaworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path
from metaworld.core.multitask_env import MultitaskEnv
from metaworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv

from pyquaternion import Quaternion
from metaworld.envs.mujoco.utils.rotation import euler2quat

from metaworld.envs.mujoco.sawyer_xyz.base import OBS_TYPE

class SawyerRandomEnv(SawyerXYZEnv):
    def __init__(
            self,
            obs_type='plain',
            goal_low=None,
            goal_high=None,
            rotMode='fixed',
            objects=None, 
            positions=None,
            state_rep='feature',
            reward_type=False,
            obj_ids=None,
            camera_pos='right',
            max_timesteps=1000,
            **kwargs
    ):
        self.quick_init(locals())
        self._objects = objects
        self._positions = positions
        self._obj_ids = obj_ids
        self._camera_pos = camera_pos
        self.state_rep = state_rep
        self.reward_type = reward_type

        obj_low=(-0.4, 0.4, 0.05)
        obj_high=(0.4, 1.0, 0.05)
        SawyerXYZEnv.__init__(
            self,
            frame_skip=5,
            action_scale=1./100,
            model_name=self.model_name,
            **kwargs
        )
        self.init_config = {
            'obj_init_pos': np.array([0, 0.7, 0.05]),
            'hand_init_pos': np.array([0, 0.3, 1.2], dtype=np.float32),
        }
        self.goal = np.array([0., 0.73, 0.08])
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.hand_init_pos = self.init_config['hand_init_pos']

        assert obs_type in OBS_TYPE
        self.obs_type = obs_type

        if goal_low is None:
            goal_low = self.hand_low
        
        if goal_high is None:
            goal_high = self.hand_high

        self.max_path_length = max_timesteps
        self.rotMode = rotMode
        if rotMode == 'fixed':
            self.action_space = Box(
                np.array([-1, -1, -1, -1]),
                np.array([1, 1, 1, 1]),
            )
        elif rotMode == 'rotz':
            self.action_rot_scale = 1./50
            self.action_space = Box(
                np.array([-1, -1, -1, -np.pi, -1]),
                np.array([1, 1, 1, np.pi, 1]),
            )
        elif rotMode == 'quat':
            self.action_space = Box(
                np.array([-1, -1, -1, 0, -1, -1, -1, -1]),
                np.array([1, 1, 1, 2*np.pi, 1, 1, 1, 1]),
            )
        else:
            self.action_space = Box(
                np.array([-1, -1, -1, -np.pi/2, -np.pi/2, 0, -1]),
                np.array([1, 1, 1, np.pi/2, np.pi/2, np.pi*2, 1]),
            )
        self.obj_and_goal_space = Box(
            np.array(obj_low),
            np.array(obj_high),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))
        if self.obs_type == 'plain':
            self.observation_space = Box(
                np.hstack((self.hand_low, obj_low,)),
                np.hstack((self.hand_high, obj_high,)),
            )
        elif self.obs_type == 'with_goal':
            self.observation_space = Box(
                np.hstack((self.hand_low, obj_low, goal_low)),
                np.hstack((self.hand_high, obj_high, goal_high)),
            )
        else:
            raise NotImplementedError
        self.reset()

    def set_positions(self, positions):
        self._positions = positions

    def get_goal(self):
        return {
            'state_desired_goal': self._state_goal,
    }

    @property
    def model_name(self):
        return get_asset_full_path('sawyer_xyz/sawyer_random.xml')

    # @profile
    def step(self, action):
        if self.rotMode == 'euler':
            action_ = np.zeros(7)
            action_[:3] = action[:3]
            action_[3:] = euler2quat(action[3:6])
            self.set_xyz_action_rot(action_)
        elif self.rotMode == 'fixed':
            self.set_xyz_action(action[:3])
        elif self.rotMode == 'rotz':
            self.set_xyz_action_rotz(action[:4])
        else:
            self.set_xyz_action_rot(action[:7])
        self.do_simulation([action[-1], -action[-1]])

        # The marker seems to get reset every time you do a simulation
        ob_feature = self._get_obs()
        self.curr_path_length +=1
        reward, success = self.compute_reward(action, ob_feature)
        if (self.curr_path_length > self.max_path_length) or success:
            done = True
        else:
            done = False
        if self.state_rep == 'feature':
            return ob_feature, reward, done, success
        elif self.state_rep == 'pixel':
            ob_pixel = self.render(mode = 'rgb_array')
            hand = self.get_endeff_pos()
            ob_pixel = (ob_pixel, hand)
            return ob_pixel, reward, done, success
        else:
            raise NotImplementedError('Invalid state representation!')

    def get_angle(self):
        return np.array([self.data.get_joint_qpos('joint')])

    def get_mocap_quat(self):
        return self.data.get_mocap_quat('mocap')

    def _get_obs(self):
        hand = self.get_endeff_pos()
        if self._obj_goal == 'button_top':
            objPos = self.data.site_xpos[self.model.site_name2id('buttonTopStart')]
        elif self._obj_goal == 'button_side':
            objPos = self.data.site_xpos[self.model.site_name2id('buttonSideStart')]
        elif self._obj_goal == 'coffee_button':
            objPos = self.data.site_xpos[self.model.site_name2id('coffeeButtonStart')]
        elif self._obj_goal == 'handle_press_top':
            objPos = self.data.site_xpos[self.model.site_name2id('handleTopStart')]
        elif self._obj_goal == 'handle_press_side':
            objPos = self.data.site_xpos[self.model.site_name2id('handleSideStart')]
        elif self._obj_goal == 'door_lock':
            objPos = self.data.site_xpos[self.model.site_name2id('doorLockStartLock')]
        elif self._obj_goal == 'door_unlock':
            objPos = self.data.site_xpos[self.model.site_name2id('doorLockStartUnlock')]
        elif self._obj_goal == 'dial_turn':
            objPos = self.data.site_xpos[self.model.site_name2id('dialStart')]
        elif self._obj_goal == 'faucet_open':
            objPos = self.data.site_xpos[self.model.site_name2id('faucetHandleStartOpen')]
        elif self._obj_goal == 'faucet_close':
            objPos = self.data.site_xpos[self.model.site_name2id('faucetHandleStartClose')]
        elif self._obj_goal == 'window_open':
            objPos = self.data.site_xpos[self.model.site_name2id('windowHandleOpenStart')]
        elif self._obj_goal == 'window_close':
            objPos = self.data.site_xpos[self.model.site_name2id('windowHandleCloseStart')]
        elif self._obj_goal == 'peg_unplug':
            objPos = self.data.site_xpos[self.model.site_name2id('pegEnd')]

        flat_obs = np.concatenate((hand, objPos))
        if self.obs_type == 'with_goal_and_id':
            return np.concatenate([
                    flat_obs,
                    self._state_goal,
                    self._state_goal_idx
                ])
        elif self.obs_type == 'with_goal':
            return np.concatenate([
                    flat_obs,
                    self._state_goal
                ])
        elif self.obs_type == 'plain':
            return np.concatenate([flat_obs,])  # TODO ZP do we need the concat?
        else:
            return np.concatenate([flat_obs, self._state_goal_idx])

    # def _get_obs(self):
    #     hand = self.get_endeff_pos()
    #     objPos =  self.data.site_xpos[self.model.site_name2id('buttonTopStart')]
    #     flat_obs = np.concatenate((hand, objPos))
    #     if self.obs_type == 'with_goal_and_id':
    #         return np.concatenate([
    #                 flat_obs,
    #                 self._state_goal,
    #                 self._state_goal_idx
    #             ])
    #     elif self.obs_type == 'with_goal':
    #         return np.concatenate([
    #                 flat_obs,
    #                 self._state_goal
    #             ])
    #     elif self.obs_type == 'plain':
    #         return np.concatenate([flat_obs,])  # TODO ZP do we need the concat?
    #     else:
    #         return np.concatenate([flat_obs, self._state_goal_idx])

    def _get_obs_dict(self):
        hand = self.get_endeff_pos()
        objPos =  self.get_site_pos('dialStart')
        flat_obs = np.concatenate((hand, objPos))
        return dict(
            state_observation=flat_obs,
            state_desired_goal=self._state_goal,
            state_achieved_goal=objPos,
        )

    def _get_info(self):
        pass

    def _set_objCOM_marker(self):
        """
        This should be use ONLY for visualization. Use self._state_goal for
        logging, learning, etc.
        """
        pass
    
    def _set_goal_marker(self, goal):
        """
        This should be use ONLY for visualization. Use self._state_goal for
        logging, learning, etc.
        """
        pass

    def _set_obj_xyz_quat(self, pos, angle):
        quat = Quaternion(axis = [0,0,1], angle = angle).elements
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9:12] = pos.copy()
        qpos[12:16] = quat.copy()
        qvel[9:15] = 0
        self.set_state(qpos, qvel)

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        joint_id = self.model.joint_names.index('doorJoint')
        qpos[joint_id] = pos
        qvel[joint_id] = 0
        self.set_state(qpos, qvel)


    def reset_model(self):
        self._reset_hand()

        self.sim.model.body_pos[self.model.body_name2id('box_top')] = np.array([-10, -10, -10.])
        self.sim.model.body_pos[self.model.body_name2id('button_top')] = np.array([-10, -9, -10.])
        self.sim.model.body_pos[self.model.body_name2id('box_side')] = np.array([-10, -8, -10.])
        self.sim.model.body_pos[self.model.body_name2id('button_side')] = np.array([-10, -7, -10.])
        self.sim.model.body_pos[self.model.body_name2id('coffee_button')] = np.array([-10, -6, -10.])
        self.sim.model.body_pos[self.model.body_name2id('coffee_machine')] = np.array([-9, -10, -10.])
        self.sim.model.body_pos[self.model.body_name2id('handle_press_top')] = np.array([-9, -9, -10.])
        self.sim.model.body_pos[self.model.body_name2id('handle_press_side')] = np.array([-9, -8, -10.])
        self.sim.model.body_pos[self.model.body_name2id('box_handle_press_top')] = np.array([-9, -7, -10.])
        self.sim.model.body_pos[self.model.body_name2id('box_handle_press_side')] = np.array([-9, -6, -10.])
        self.sim.model.body_pos[self.model.body_name2id('door')] = np.array([-8, -10, -10.])
        self.sim.model.body_pos[self.model.body_name2id('door_lock')] = np.array([-8, -9, -10.])
        self.sim.model.body_pos[self.model.body_name2id('dial')] = np.array([-8, -8, -10.])
        self.sim.model.body_pos[self.model.body_name2id('faucetBase')] = np.array([-8, -7, -10.])
        self.sim.model.body_pos[self.model.body_name2id('faucet')] = np.array([-8, -6, -10.])
        self.sim.model.body_pos[self.model.body_name2id('window')] = np.array([-7, -10, -10.])
        self.sim.model.body_pos[self.model.body_name2id('window_another')] = np.array([-7, -9, -10.])
        self.sim.model.body_pos[self.model.body_name2id('window_wall')] = np.array([-7, -8, -10.])
        self.sim.model.body_pos[self.model.body_name2id('box')] = np.array([-7, -7, -10.])
        self.sim.model.body_pos[self.model.body_name2id('peg')] = np.array([-7, -6, -10.])
        # randomly sample objects for each position
        self._state_goal = None
        self._obj_goal = None
        for obj_id, pos in zip(self._obj_ids, self._positions):
            (x_, y_) = pos
            x = 0.2*x_ - 0.4
            y = 0.1*y_ + 0.5
            curr_obj = self._objects[obj_id]
            if curr_obj == 'button_top':
                box_pos = np.array([x, y, 0.05])
                button_pos = box_pos.copy()
                button_pos[1] += 0.08
                button_pos[2] += 0.07
                self.sim.model.body_pos[self.model.body_name2id('box_top')] = box_pos
                self.sim.model.body_pos[self.model.body_name2id('button_top')] = button_pos
                if self._state_goal is None:
                    self._obj_goal = 'button_top'
                    self._state_goal = button_pos.copy()
                    self._state_goal[2] -= 0.02
                self.maxDist = np.abs(self.data.site_xpos[self.model.site_name2id('buttonTopStart')][2] - self._state_goal[2])
            elif curr_obj == 'button_side':
                box_pos = np.array([x, y, 0.05])
                button_pos = box_pos.copy()
                button_pos[1] -= 0.06
                button_pos[2] += 0.07
                self.sim.model.body_pos[self.model.body_name2id('box_side')] = box_pos
                self.sim.model.body_pos[self.model.body_name2id('button_side')] = button_pos
                if self._state_goal is None:
                    self._obj_goal = 'button_side'
                    self._state_goal = self.get_site_pos('box_side_hole')
                self.maxDist = np.abs(self.data.site_xpos[self.model.site_name2id('buttonSideStart')][1] - self._state_goal[1])
            elif curr_obj == 'coffee_button':
                coffee_machine_pos = np.array([x, y, 0.28])
                button_pos = coffee_machine_pos + np.array([0., -0.12, 0.05])
                self.sim.model.body_pos[self.model.body_name2id('coffee_machine')] = coffee_machine_pos
                self.sim.model.body_pos[self.model.body_name2id('coffee_button')] = button_pos
                if self._state_goal is None:
                    self._obj_goal = 'coffee_button'
                    self._state_goal = self.get_site_pos('coffee_goal')
                self.maxDist = np.abs(self.data.site_xpos[self.model.site_name2id('coffeeButtonStart')][1] - self._state_goal[1])
            elif curr_obj == 'handle_press_top':
                box_handle_press_top_pos = np.array([x, y, 0.05])
                handle_press_top_pos = box_handle_press_top_pos.copy()
                handle_press_top_pos[1] -= 0.1
                handle_press_top_pos[2] += 0.09
                self.sim.model.body_pos[self.model.body_name2id('handle_press_top')] = handle_press_top_pos
                self.sim.model.body_pos[self.model.body_name2id('box_handle_press_top')] = box_handle_press_top_pos
                if self._state_goal is None:
                    self._obj_goal = 'handle_press_top'
                    self._state_goal = self.get_site_pos('handleTopGoalPress')
                self.maxDist = np.abs(self.data.site_xpos[self.model.site_name2id('handleTopStart')][-1] - self._state_goal[-1])
            elif curr_obj == 'handle_press_side':
                box_handle_press_side_pos = np.array([x, y, 0.05])
                handle_press_side_pos = box_handle_press_side_pos.copy()
                handle_press_side_pos[0] += 0.1
                handle_press_side_pos[2] += 0.09
                self.sim.model.body_pos[self.model.body_name2id('handle_press_side')] = handle_press_side_pos
                self.sim.model.body_pos[self.model.body_name2id('box_handle_press_side')] = box_handle_press_side_pos
                if self._state_goal is None:
                    self._obj_goal = 'handle_press_side'
                    self._state_goal = self.get_site_pos('handleSideGoalPress')
                self.maxDist = np.abs(self.data.site_xpos[self.model.site_name2id('handleSideStart')][-1] - self._state_goal[-1])
            elif curr_obj == 'door_lock':
                door_pos = np.array([x, y, 0.1])
                self.sim.model.body_pos[self.model.body_name2id('door')] = door_pos
                self.sim.model.body_pos[self.model.body_name2id('door_lock')] = door_pos
                if self._state_goal is None:
                    self._obj_goal = 'door_lock'
                    self._state_goal = door_pos + np.array([0, -0.04, -0.03])
                    self.maxPullDist = np.linalg.norm(self._state_goal - self.data.get_geom_xpos('doorLockGeom'))
            elif curr_obj == 'door_unlock':
                door_pos = np.array([x, y, 0.1])
                self.sim.model.body_pos[self.model.body_name2id('door')] = door_pos
                self.sim.model.body_pos[self.model.body_name2id('door_lock')] = door_pos
                # rotate the handle
                self._set_obj_xyz(1.5708)
                if self._state_goal is None:
                    self._obj_goal = 'door_unlock'
                    self._state_goal = door_pos + np.array([0.1, -0.04, 0.07])
                    self.maxPullDist = np.linalg.norm(self._state_goal - self.data.get_geom_xpos('doorLockGeom'))
            elif curr_obj == 'dial_turn':
                dial_pos = np.array([x, y, 0.05])
                self.sim.model.body_pos[self.model.body_name2id('dial')] = dial_pos
                if self._state_goal is None:
                    self._obj_goal = 'dial_turn'
                    self._state_goal = dial_pos + np.array([0, 0.03, 0.03])
                    self.maxPullDist = np.abs(self._state_goal[1] - y)
            elif curr_obj == 'faucet_open':
                faucet_pos = np.array([x, y, 0.05])
                self.sim.model.body_pos[self.model.body_name2id('faucet')] = faucet_pos
                self.sim.model.body_pos[self.model.body_name2id('faucetBase')] = faucet_pos
                if self._state_goal is None:
                    self._obj_goal = 'faucet_open'
                    self._state_goal = faucet_pos + np.array([0.1, -0.015, 0.065])
                    self.maxPullDist = np.linalg.norm(self._state_goal - faucet_pos)
            elif curr_obj == 'faucet_close':
                faucet_pos = np.array([x, y, 0.05])
                self.sim.model.body_pos[self.model.body_name2id('faucet')] = faucet_pos
                self.sim.model.body_pos[self.model.body_name2id('faucetBase')] = faucet_pos
                if self._state_goal is None:
                    self._obj_goal = 'faucet_close'
                    self._state_goal = faucet_pos + np.array([-0.1, -0.015, 0.065])
                    self.maxPullDist = np.linalg.norm(self._state_goal - faucet_pos)

            elif curr_obj == 'window_open':
                window_pos = np.array([x, y, 0.16])
                window_another_pos = window_pos + np.array([0.2, 0.03, 0])
                wall_pos = window_pos - np.array([-0.1, 0, 0.12])
                self.sim.model.body_pos[self.model.body_name2id('window')] = window_pos
                self.sim.model.body_pos[self.model.body_name2id('window_another')] = window_another_pos
                self.sim.model.body_pos[self.model.body_name2id('window_wall')] = wall_pos
                if self._state_goal is None:
                    self._obj_goal = 'window_open'
                    self._state_goal = window_pos.copy()
                    self._state_goal[0] += 0.18
                self.liftThresh = 0.02
                self.objHeight = self.data.get_geom_xpos('windowHandle')[2]
                self.heightTarget = self.objHeight + self.liftThresh
                self.maxPullDist = 0.2
                self.maxPullDist = 0.2
            elif curr_obj == 'window_close':
                window_pos = np.array([x, y, 0.16])
                window_another_pos = window_pos + np.array([0, 0.03, 0])
                wall_pos = window_pos - np.array([0.1, 0, 0.12])
                self.sim.model.body_pos[self.model.body_name2id('window')] = window_pos
                self.sim.model.body_pos[self.model.body_name2id('window_another')] = window_another_pos
                self.sim.model.body_pos[self.model.body_name2id('window_wall')] = wall_pos
                if self._state_goal is None:
                    self._obj_goal = 'window_close'
                    self._state_goal = window_pos.copy()
                    self._state_goal[0] -= 0.18
                self.liftThresh = 0.02
                self.objHeight = self.data.get_geom_xpos('windowHandle')[2]
                self.heightTarget = self.objHeight + self.liftThresh
                self.maxPullDist = 0.2
            elif curr_obj == 'peg_unplug':
                box_pos = np.array([x, y, 0.05])
                self.sim.model.body_pos[self.model.body_name2id('box')] = box_pos
                hole_pos = self.sim.model.site_pos[self.model.site_name2id('hole')] + self.sim.model.body_pos[self.model.body_name2id('box')]
                self.sim.model.body_pos[self.model.body_name2id('peg')] = hole_pos
                if self._state_goal is None:
                    self._obj_goal = 'peg_unplug'
                    self._state_goal = hole_pos.copy()
                    self._state_goal[0] += 0.2
                self.liftThresh = 0.04
                self.objHeight = self.get_body_com('peg').copy()[0]
                self.heightTarget = self.objHeight + self.liftThresh
                self.maxPlacingDist = np.linalg.norm(self._state_goal - self.get_body_com('peg'))


        self.curr_path_length = 0
        if self.state_rep == 'feature':
            return self._get_obs()
        elif self.state_rep == 'pixel':
            ob_pixel = self.render(mode = 'rgb_array')
            hand = self.get_endeff_pos()
            ob_pixel = (ob_pixel, hand)
            return ob_pixel
        else:
            raise NotImplementedError('Invalid state representation!')

    def _reset_hand(self):
        hand_low = (-0.1, 0.3, 0.8)
        hand_high = (0.1, 0.4, 1.2)
        hand_pos = np.random.uniform(hand_low, hand_high, size=3)

        for i in range(10):
            self.data.set_mocap_pos('mocap', hand_pos)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            self.do_simulation([-1, 1], self.frame_skip)

        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        self.init_fingerCOM  =  (rightFinger + leftFinger)/2
        self.reachCompleted = False

    def get_site_pos(self, siteName):
        _id = self.model.site_names.index(siteName)
        return self.data.site_xpos[_id].copy()

    def compute_rewards(self, actions, obsBatch):
        #Required by HER-TD3
        assert isinstance(obsBatch, dict) == True
        obsList = obsBatch['state_observation']
        rewards = [self.compute_reward(action, obs)[0] for  action, obs in zip(actions, obsList)]
        return np.array(rewards)

    def compute_reward(self, actions, obs):
        if self._obj_goal == 'button_top':
            if isinstance(obs, dict): 
                obs = obs['state_observation']

            objPos = obs[3:6]

            rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
            fingerCOM  =  (rightFinger + leftFinger)/2

            pressGoal = self._state_goal[2]

            pressDist = np.abs(objPos[2] - pressGoal)
            reachDist = np.linalg.norm(objPos - fingerCOM)
            reachDistxy = np.linalg.norm(objPos[:-1] - fingerCOM[:-1])
            reachDistxyz = np.linalg.norm(np.concatenate((objPos[:-1], [self.init_fingerCOM[-1]])) - fingerCOM)
            reachRew = -reachDist
            c1 = 1000 ; c2 = 0.01 ; c3 = 0.001
            if reachDist < 0.05:
                pressRew = 1000*(self.maxDist - pressDist) + c1*(np.exp(-(pressDist**2)/c2) + np.exp(-(pressDist**2)/c3))
            else:
                pressRew = 0
            pressRew = max(pressRew, 0)
            reward = reachRew + pressRew
            success = (pressDist <= 0.02)

        elif self._obj_goal == 'button_side':
            if isinstance(obs, dict): 
                obs = obs['state_observation']

            objPos = obs[3:6]

            leftFinger = self.get_site_pos('leftEndEffector')
            fingerCOM  =  leftFinger

            pressGoal = self._state_goal[1]

            pressDist = np.abs(objPos[1] - pressGoal)
            reachDist = np.linalg.norm(objPos - fingerCOM)
            c1 = 1000 ; c2 = 0.01 ; c3 = 0.001
            if reachDist < 0.05:
                pressRew = 1000*(self.maxDist - pressDist) + c1*(np.exp(-(pressDist**2)/c2) + np.exp(-(pressDist**2)/c3))
            else:
                pressRew = 0
            pressRew = max(pressRew, 0)
            reward = -reachDist + pressRew
            success = (pressDist <= 0.02)

        elif self._obj_goal == 'coffee_button':
            if isinstance(obs, dict): 
                obs = obs['state_observation']

            objPos = obs[3:6]

            leftFinger = self.get_site_pos('leftEndEffector')
            fingerCOM  =  leftFinger

            pressGoal = self._state_goal[1]

            pressDist = np.abs(objPos[1] - pressGoal)
            reachDist = np.linalg.norm(objPos - fingerCOM)
            c1 = 1000 ; c2 = 0.01 ; c3 = 0.001
            if reachDist < 0.05:
                pressRew = 1000*(self.maxDist - pressDist) + c1*(np.exp(-(pressDist**2)/c2) + np.exp(-(pressDist**2)/c3))
            else:
                pressRew = 0
            pressRew = max(pressRew, 0)
            reward = -reachDist + pressRew
            success = (pressDist <= 0.02)

        elif self._obj_goal == 'handle_press_top':
            if isinstance(obs, dict): 
                obs = obs['state_observation']

            objPos = obs[3:6]

            leftFinger = self.get_site_pos('leftEndEffector')
            fingerCOM  =  leftFinger

            pressGoal = self._state_goal[-1]

            pressDist = np.abs(objPos[-1] - pressGoal)
            reachDist = np.linalg.norm(objPos - fingerCOM)
            c1 = 1000 ; c2 = 0.01 ; c3 = 0.001
            if reachDist < 0.05:
                pressRew = 1000*(self.maxDist - pressDist) + c1*(np.exp(-(pressDist**2)/c2) + np.exp(-(pressDist**2)/c3))
            else:
                pressRew = 0
            pressRew = max(pressRew, 0)
            reward = -reachDist + pressRew
            success = (pressDist <= 0.04)

        elif self._obj_goal == 'handle_press_side':
            if isinstance(obs, dict): 
                obs = obs['state_observation']

            objPos = obs[3:6]

            leftFinger = self.get_site_pos('leftEndEffector')
            fingerCOM  =  leftFinger

            pressGoal = self._state_goal[-1]

            pressDist = np.abs(objPos[-1] - pressGoal)
            reachDist = np.linalg.norm(objPos - fingerCOM)
            c1 = 1000 ; c2 = 0.01 ; c3 = 0.001
            if reachDist < 0.05:
                pressRew = 1000*(self.maxDist - pressDist) + c1*(np.exp(-(pressDist**2)/c2) + np.exp(-(pressDist**2)/c3))
            else:
                pressRew = 0
            pressRew = max(pressRew, 0)
            reward = -reachDist + pressRew
            success = (pressDist <= 0.04)

        elif self._obj_goal == 'door_lock':
            if isinstance(obs, dict): 
                obs = obs['state_observation']

            objPos = obs[3:6]

            rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
            fingerCOM  =  (rightFinger + leftFinger)/2

            pullGoal = self._state_goal

            pullDist = np.linalg.norm(objPos - pullGoal)
            reachDist = np.linalg.norm(objPos - fingerCOM)
            reachRew = -reachDist

            def reachCompleted():
                if reachDist < 0.05:
                    return True
                else:
                    return False

            if reachCompleted():
                self.reachCompleted = True
            else:
                self.reachCompleted = False

            def pullReward():
                c1 = 1000 ; c2 = 0.01 ; c3 = 0.001
                if self.reachCompleted:
                    pullRew = 1000*(self.maxPullDist - pullDist) + c1*(np.exp(-(pullDist**2)/c2) + np.exp(-(pullDist**2)/c3))
                    pullRew = max(pullRew,0)
                    return pullRew
                else:
                    return 0
            pullRew = pullReward()
            reward = reachRew + pullRew# - actions[-1]/50
            success = (pullDist <= 0.04)
        elif self._obj_goal == 'door_unlock':
            if isinstance(obs, dict): 
                obs = obs['state_observation']

            objPos = obs[3:6]

            rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
            fingerCOM  =  (rightFinger + leftFinger)/2

            pullGoal = self._state_goal

            pullDist = np.linalg.norm(objPos - pullGoal)
            reachDist = np.linalg.norm(objPos - fingerCOM)
            reachRew = -reachDist

            def reachCompleted():
                if reachDist < 0.05:
                    return True
                else:
                    return False

            if reachCompleted():
                self.reachCompleted = True

            def pullReward():
                c1 = 1000 ; c2 = 0.01 ; c3 = 0.001
                if self.reachCompleted:
                    pullRew = 1000*(self.maxPullDist - pullDist) + c1*(np.exp(-(pullDist**2)/c2) + np.exp(-(pullDist**2)/c3))
                    pullRew = max(pullRew,0)
                    return pullRew
                else:
                    return 0
            pullRew = pullReward()
            reward = reachRew + pullRew# - actions[-1]/50
            success = (pullDist <= 0.05)
          
        elif self._obj_goal == 'dial_turn':
            if isinstance(obs, dict): 
                obs = obs['state_observation']

            objPos = obs[3:6]

            rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
            fingerCOM  =  (rightFinger + leftFinger)/2

            pullGoal = self._state_goal

            pullDist = np.abs(objPos[1] - pullGoal[1])# + np.abs(objPos[0] - pullGoal[0])
            reachDist = np.linalg.norm(objPos - fingerCOM)
            reachRew = -reachDist

            def reachCompleted():
                if reachDist < 0.05:
                    return True
                else:
                    return False

            if reachCompleted():
                self.reachCompleted = True
            else:
                self.reachCompleted = False

            def pullReward():
                c1 = 1000 ; c2 = 0.001 ; c3 = 0.0001
                if self.reachCompleted:
                    pullRew = 1000*(self.maxPullDist - pullDist) + c1*(np.exp(-(pullDist**2)/c2) + np.exp(-(pullDist**2)/c3))
                    pullRew = max(pullRew,0)
                    return pullRew
                else:
                    return 0
            pullRew = pullReward()
            reward = reachRew + pullRew# - actions[-1]/50
            success = (pullDist <= 0.03)
        
        elif self._obj_goal == 'faucet_open':
            if isinstance(obs, dict): 
                obs = obs['state_observation']

            objPos = obs[3:6]

            rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
            fingerCOM  =  (rightFinger + leftFinger)/2

            pullGoal = self._state_goal

            pullDist = np.linalg.norm(objPos - pullGoal)
            reachDist = np.linalg.norm(objPos - fingerCOM)  
            reachRew = -reachDist

            def reachCompleted():
                if reachDist < 0.05:
                    return True
                else:
                    return False

            if reachCompleted():
                self.reachCompleted = True
            else:
                self.reachCompleted = False

            def pullReward():
                c1 = 1000 ; c2 = 0.01 ; c3 = 0.001
                if self.reachCompleted:
                    pullRew = 1000*(self.maxPullDist - pullDist) + c1*(np.exp(-(pullDist**2)/c2) + np.exp(-(pullDist**2)/c3))
                    pullRew = max(pullRew,0)
                    return pullRew
                else:
                    return 0
            pullRew = pullReward()
            reward = reachRew + pullRew# - actions[-1]/50
            success = (pullDist <= 0.05)
          
        elif self._obj_goal == 'faucet_close':
            if isinstance(obs, dict): 
                obs = obs['state_observation']

            objPos = obs[3:6]

            rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
            fingerCOM  =  (rightFinger + leftFinger)/2

            pullGoal = self._state_goal

            pullDist = np.linalg.norm(objPos - pullGoal)
            reachDist = np.linalg.norm(objPos - fingerCOM)
            reachRew = -reachDist

            def reachCompleted():
                if reachDist < 0.05:
                    return True
                else:
                    return False

            if reachCompleted():
                self.reachCompleted = True
            else:
                self.reachCompleted = False

            def pullReward():
                c1 = 1000 ; c2 = 0.01 ; c3 = 0.001
                if self.reachCompleted:
                    pullRew = 1000*(self.maxPullDist - pullDist) + c1*(np.exp(-(pullDist**2)/c2) + np.exp(-(pullDist**2)/c3))
                    pullRew = max(pullRew,0)
                    return pullRew
                else:
                    return 0
            pullRew = pullReward()
            reward = reachRew + pullRew# - actions[-1]/50
            success = (pullDist <= 0.05)
          
        elif self._obj_goal == 'window_open':
            if isinstance(obs, dict): 
                obs = obs['state_observation']

            objPos = obs[3:6]

            rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
            fingerCOM  =  (rightFinger + leftFinger)/2

            pullGoal = self._state_goal

            pullDist = np.abs(objPos[0] - pullGoal[0])
            reachDist = np.linalg.norm(objPos - fingerCOM)
            heightTarget = self.heightTarget

            def reachCompleted():
                if reachDist < 0.05:
                    return True
                else:
                    return False

            if reachCompleted():
                self.reachCompleted = True
            else:
                self.reachCompleted = False

            c1 = 1000 ; c2 = 0.01 ; c3 = 0.001
            reachRew = -reachDist
            if self.reachCompleted:
                pullRew = 1000*(self.maxPullDist - pullDist) + c1*(np.exp(-(pullDist**2)/c2) + np.exp(-(pullDist**2)/c3))
            else:
                pullRew = 0
            reward = reachRew + pullRew
            success = (pullDist <= 0.05)
        
        elif self._obj_goal == 'window_close':
            if isinstance(obs, dict): 
                obs = obs['state_observation']

            objPos = obs[3:6]

            rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
            fingerCOM  =  (rightFinger + leftFinger)/2

            pullGoal = self._state_goal

            pullDist = np.abs(objPos[0] - pullGoal[0])
            reachDist = np.linalg.norm(objPos - fingerCOM)
            heightTarget = self.heightTarget

            def reachCompleted():
                if reachDist < 0.05:
                    return True
                else:
                    return False

            if reachCompleted():
                self.reachCompleted = True
            else:
                self.reachCompleted = False

            c1 = 1000 ; c2 = 0.01 ; c3 = 0.001
            reachRew = -reachDist
            if self.reachCompleted:
                pullRew = 1000*(self.maxPullDist - pullDist) + c1*(np.exp(-(pullDist**2)/c2) + np.exp(-(pullDist**2)/c3))
            else:
                pullRew = 0
            reward = reachRew + pullRew
            success = (pullDist <= 0.05)
        
        elif self._obj_goal == 'peg_unplug':
            if isinstance(obs, dict):
                obs = obs['state_observation']

            objPos = obs[3:6]
            pegHeadPos = self.get_site_pos('pegHead')

            rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
            fingerCOM  =  (rightFinger + leftFinger)/2

            heightTarget = self.heightTarget
            placingGoal = self._state_goal

            reachDist = np.linalg.norm(objPos - fingerCOM)

            placingDist = np.linalg.norm(objPos[:-1] - placingGoal[:-1])
          

            def reachReward():
                reachDistxy = np.linalg.norm(objPos[:-1] - fingerCOM[:-1])
                zRew = np.linalg.norm(fingerCOM[-1] - self.hand_init_pos[-1])
                if reachDistxy < 0.05: #0.02
                    reachRew = -reachDist
                else:
                    reachRew =  -reachDistxy - 2*zRew
                #incentive to close fingers when reachDist is small
                if reachDist < 0.05:
                    reachRew = -reachDist + max(actions[-1],0)/50
                return reachRew, reachDist

            def reachCompleted():
                if reachDist < 0.05:
                    return True
                else:
                    return False

            if reachCompleted():
                self.reachCompleted = True

            def placeReward():
                c1 = 1000 ; c2 = 0.01 ; c3 = 0.001
                if self.reachCompleted:
                    placeRew = 1000*(self.maxPlacingDist - placingDist) + c1*(np.exp(-(placingDist**2)/c2) + np.exp(-(placingDist**2)/c3))
                    placeRew = max(placeRew,0)
                    return [placeRew , placingDist]
                else:
                    return [0 , placingDist]

            reachRew, reachDist = reachReward()
            placeRew , placingDist = placeReward()
            assert placeRew >=0
            reward = reachRew + placeRew
            success = (placingDist <= 0.07)

        if self.reward_type == 'sparse':
            reward = 1 if success else 0
        return reward, success

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        return statistics

    def log_diagnostics(self, paths = None, logger = None):
        pass

    def get_frame(self):
        hand = self.get_endeff_pos()
        self._camera_pos = 'left'
        obs_pixel_left = self.render(mode = 'rgb_array')
        self._camera_pos = 'center'
        obs_pixel_center = self.render(mode = 'rgb_array')
        self._camera_pos = 'right'
        obs_pixel_right = self.render(mode = 'rgb_array')
        obs = (obs_pixel_left, obs_pixel_center, obs_pixel_right, None)
        return obs

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.lookat[0] = 0
        self.viewer.cam.lookat[1] = 0.75
        self.viewer.cam.lookat[2] = 0.4
        self.viewer.cam.distance = 2.0
        self.viewer.cam.elevation = -30
        self.viewer.cam.trackbodyid = -1
        if self._camera_pos == 'right':
            self.viewer.cam.azimuth = 135
        elif self._camera_pos == 'center':
            self.viewer.cam.azimuth = 90
            self.viewer.cam.elevation = -30
            self.viewer.cam.distance = 1.10
        elif self._camera_pos == 'left':
            self.viewer.cam.azimuth = 45

