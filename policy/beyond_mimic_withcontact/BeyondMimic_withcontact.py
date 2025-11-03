from common.path_config import PROJECT_ROOT

from FSM.FSMState import FSMStateName, FSMState
from common.ctrlcomp import StateAndCmd, PolicyOutput
import numpy as np
import yaml
from common.utils import FSMCommand, progress_bar
import onnx
import onnxruntime
import torch
import os
from typing import Literal

class BeyondMimic_Contact(FSMState):
    def __init__(self, state_cmd:StateAndCmd, policy_output:PolicyOutput):
        super().__init__()
        self.state_cmd = state_cmd
        self.policy_output = policy_output
        self.name = FSMStateName.SKILL_BEYOND_MIMIC
        self.name_str = "beyond_mimic_withcontact"
        self.motion_phase = 0
        self.counter_step = 0
        self.ref_motion_phase = 0
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config", "BeyondMimic.yaml")
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            self.onnx_path = os.path.join(current_dir, "model", config["onnx_path"])
            self.kps_lab = np.array(config["kp_lab"], dtype=np.float32)
            self.kds_lab = np.array(config["kd_lab"], dtype=np.float32)
            self.default_angles_lab =  np.array(config["default_angles_lab"], dtype=np.float32)
            self.mj2lab =  np.array(config["mj2lab"], dtype=np.int32)
            self.tau_limit =  np.array(config["tau_limit"], dtype=np.float32)
            self.num_actions = config["num_actions"]
            self.num_obs = config["num_obs"]
            self.action_scale_lab = np.array(config["action_scale_lab"], dtype=np.float32)
            self.motion_length = config["motion_length"]
            self.mode: Literal["baseline", "saferl"] = config["mode"]
            
            self.qj_obs = np.zeros(self.num_actions, dtype=np.float32)
            self.dqj_obs = np.zeros(self.num_actions, dtype=np.float32)
            self.obs = np.zeros(self.num_obs)
            self.action = np.zeros(self.num_actions)
            
            self.ref_joint_pos = np.zeros(self.num_actions, dtype=np.float32)
            self.ref_joint_vel = np.zeros(self.num_actions, dtype=np.float32)
            self.ref_body_pos_w = np.zeros((1, 14, 3), dtype=np.float32)
            self.ref_body_quat_w = np.zeros((1, 14, 4), dtype=np.float32)
            self.ref_body_lin_vel_w = np.zeros((1, 14, 3), dtype=np.float32)
            self.ref_body_ang_vel_w = np.zeros((1, 14, 3), dtype=np.float32)
            # -- for contact
            self.ref_contact = np.zeros(2, dtype=bool)
            
            # load policy
            self.onnx_model = onnx.load(self.onnx_path)
            self.ort_session = onnxruntime.InferenceSession(self.onnx_path)
            input = self.ort_session.get_inputs()
            self.input_name = []
            for i, inpt in enumerate(input):
                self.input_name.append(inpt.name)
                
            self.onnx_metadata = {prop.key: prop.value for prop in self.onnx_model.metadata_props}
            if "time_step_total" in self.onnx_metadata:
                self.time_step_total = int(self.onnx_metadata["time_step_total"])
            else:
                self.time_step_total = self.motion_length

            print("BeyondMimic-like policy initializing ...")
    
    def enter(self):
        self.ref_motion_phase = 0.
        self.motion_time = 0
        self.counter_step = 0

        observation = {}
        observation[self.input_name[0]] = np.zeros((1, self.num_obs), dtype=np.float32)
        observation[self.input_name[1]] = np.zeros((1, 1), dtype=np.float32)
        outputs_result = self.ort_session.run(None, observation)
        # 处理多个输出
        self.action, self.ref_joint_pos, self.ref_joint_vel, _, self.ref_body_quat_w, _, _, self.ref_contact = outputs_result

        self.qj_obs = np.zeros(self.num_actions, dtype=np.float32)
        self.dqj_obs = np.zeros(self.num_actions, dtype=np.float32)
        self.obs = np.zeros(self.num_obs)

        # self.action = np.zeros(self.num_actions)

        pass
        
    def quat_mul(self, q1, q2):
        w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
        w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
        # perform multiplication
        ww = (z1 + x1) * (x2 + y2)
        yy = (w1 - y1) * (w2 + z2)
        zz = (w1 + y1) * (w2 - z2)
        xx = ww + yy + zz
        qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
        w = qq - ww + (z1 - y1) * (y2 - z2)
        x = qq - xx + (x1 + w1) * (x2 + w2)
        y = qq - yy + (w1 - x1) * (y2 + z2)
        z = qq - zz + (z1 + y1) * (w2 - x2)
        return np.array([w, x, y, z])
        
    def matrix_from_quat(self, q):
        w, x, y, z = q
        return np.array([
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)]
        ])

    def yaw_quat(self, q):
        w, x, y, z = q
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
        return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    
    def euler_single_axis_to_quat(self, angle, axis, degrees=False):
        """
        将单个欧拉角转换为四元数
        
        参数:
            angle: 旋转角度
            axis: 旋转轴，可以是 'x', 'y', 'z' 或者单位向量 [x, y, z]
            degrees: 如果为True，输入角度为度数；如果为False，输入角度为弧度
        
        返回:
            四元数 (w, x, y, z)
        """
        # 转换角度为弧度
        if degrees:
            angle = np.radians(angle)
        
        # 计算半角
        half_angle = angle * 0.5
        cos_half = np.cos(half_angle)
        sin_half = np.sin(half_angle)
        
        # 根据旋转轴确定四元数分量
        if isinstance(axis, str):
            if axis.lower() == 'x':
                return np.array([cos_half, sin_half, 0.0, 0.0])
            elif axis.lower() == 'y':
                return np.array([cos_half, 0.0, sin_half, 0.0])
            elif axis.lower() == 'z':
                return np.array([cos_half, 0.0, 0.0, sin_half])
            else:
                raise ValueError("axis must be 'x', 'y', 'z' or a 3D unit vector")
        else:
            # 假设axis是一个3D向量 [x, y, z]
            axis = np.array(axis, dtype=np.float32)
            # 归一化轴向量
            axis_norm = np.linalg.norm(axis)
            if axis_norm == 0:
                raise ValueError("axis vector cannot be zero")
            axis = axis / axis_norm
            
            # 计算四元数分量
            w = cos_half
            x = sin_half * axis[0]
            y = sin_half * axis[1]
            z = sin_half * axis[2]
            
            return np.array([w, x, y, z])

    def run(self):
        robot_quat = self.state_cmd.base_quat
        
        qj = self.state_cmd.q[self.mj2lab]
        qj = (qj - self.default_angles_lab)

        base_troso_yaw = qj[2]
        base_troso_roll = qj[5]
        base_troso_pitch = qj[8]
        
        # beyond mimic使用torso姿态作为姿态输入，需要根据腰部位置将pelvis数据转到torso
        quat_yaw = self.euler_single_axis_to_quat(base_troso_yaw, 'z', degrees=False)
        quat_roll = self.euler_single_axis_to_quat(base_troso_roll, 'x', degrees=False)
        quat_pitch = self.euler_single_axis_to_quat(base_troso_pitch, 'y', degrees=False)
        temp1 = self.quat_mul(quat_roll, quat_pitch)
        temp2 = self.quat_mul(quat_yaw, temp1)
        robot_quat = self.quat_mul(robot_quat, temp2)
        ref_anchor_ori_w = self.ref_body_quat_w[:, 7].squeeze(0)

        # 在第一帧提取当前机器人yaw方向，与参考动作yaw方向做差（与beyond mimic一致）
        if(self.counter_step < 2):
            init_to_anchor = self.matrix_from_quat(self.yaw_quat(ref_anchor_ori_w))
            world_to_anchor = self.matrix_from_quat(self.yaw_quat(robot_quat))
            self.init_to_world = world_to_anchor @ init_to_anchor.T
            print("self.init_to_world: ", self.init_to_world)
            self.counter_step += 1
            return

        motion_anchor_ori_b = self.matrix_from_quat(robot_quat).T @ self.init_to_world @ self.matrix_from_quat(ref_anchor_ori_w)

        ang_vel = self.state_cmd.ang_vel
        
        dqj = self.state_cmd.dq
        
        mimic_obs_buf = np.concatenate((self.ref_joint_pos.squeeze(0),
                                        self.ref_joint_vel.squeeze(0),
                                        motion_anchor_ori_b[:,:2].reshape(-1),
                                        self.ref_contact.squeeze(0),
                                        ang_vel,
                                        qj,
                                        dqj[self.mj2lab],
                                        self.action.squeeze(0),
                                        ),
                                        axis=-1, dtype=np.float32)
        
        mimic_obs_tensor = torch.from_numpy(mimic_obs_buf).unsqueeze(0).cpu().numpy()
        observation = {}

        # obs0 是网络观测，obs1 是当前时间步，用于输出参考动作信息
        observation[self.input_name[0]] = mimic_obs_tensor
        observation[self.input_name[1]] = np.array([[self.counter_step]], dtype=np.float32)
        outputs_result = self.ort_session.run(None, observation)

        # 处理多个输出
        self.action, self.ref_joint_pos, self.ref_joint_vel, _, self.ref_body_quat_w, _, _, self.ref_contact = outputs_result
        target_dof_pos_mj = np.zeros(29)
        target_dof_pos_lab = self.action * self.action_scale_lab + self.default_angles_lab
        target_dof_pos_mj[self.mj2lab] = target_dof_pos_lab.squeeze(0)
        
        self.policy_output.actions = target_dof_pos_mj
        self.policy_output.kps[self.mj2lab] = self.kps_lab
        self.policy_output.kds[self.mj2lab] = self.kds_lab
        
        # update motion phase
        self.counter_step += 1
        
        if self.counter_step >= self.time_step_total - 1:
            self.state_cmd.skill_cmd = FSMCommand.LOCO

    def exit(self):
        self.action = np.zeros(23, dtype=np.float32)
        # self.action_buf = np.zeros(23 * self.history_length, dtype=np.float32)
        self.ref_motion_phase = 0.
        # self.ref_motion_phase_buf = np.zeros(1 * self.history_length, dtype=np.float32)
        self.motion_time = 0
        self.counter_step = 0
        
        print("exited")

    
    def checkChange(self):
        if(self.state_cmd.skill_cmd == FSMCommand.LOCO):
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.SKILL_COOLDOWN
        elif(self.state_cmd.skill_cmd == FSMCommand.PASSIVE):
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.PASSIVE
        elif(self.state_cmd.skill_cmd == FSMCommand.POS_RESET):
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.FIXEDPOSE
        else:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.SKILL_BEYOND_MIMIC