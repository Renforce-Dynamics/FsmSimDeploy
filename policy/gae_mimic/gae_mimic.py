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


class GAE_Mimic(FSMState):
    def __init__(self, state_cmd:StateAndCmd, policy_output:PolicyOutput):
        super().__init__()
        self.state_cmd = state_cmd
        self.policy_output = policy_output
        self.name = FSMStateName.SKILL_GAE_MIMIC
        self.name_str = "gae_mimic"
        self.counter_step = 0
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config", "GAE_Mimic.yaml")
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            self.onnx_path = os.path.join(current_dir, "model", config["onnx_path"])
            self.motion_path = os.path.join(current_dir, "motion", config["motion_path"])
            self._load_motion()

            # load policy
            self.onnx_model = onnx.load(self.onnx_path)
            self.ort_session = onnxruntime.InferenceSession(self.onnx_path)
            print("Loaded GaeMimic ONNX model from ", self.onnx_path)
            
            input = self.ort_session.get_inputs()
            self.input_name = []
            for i, inpt in enumerate(input):
                self.input_name.append(inpt.name)
            print("Input names: ", self.input_name)
            
            output = self.ort_session.get_outputs()
            self.output_name = []
            for i, outpt in enumerate(output):
                self.output_name.append(outpt.name)
            print("Output names: ", self.output_name)
            
            self.metadata = self._load_metadata()
            self.kps_lab = self.metadata["joint_stiffness"]
            self.kds_lab = self.metadata["joint_damping"]
            self.default_angles_lab = self.metadata["default_joint_pos"]
            
            self.lab_joint_names = self.metadata["joint_names"]
            self.lab_body_names = self.metadata["body_names"]
            
            mj_joint_names = config["mj_joint_names"]
            self.mj2lab = [mj_joint_names.index(joint) for joint in self.lab_joint_names]

            self.motion_anchor_body_name = self.metadata["motion_anchor_body_name"]
            self.motion_anchor_id = self.lab_body_names.index(self.motion_anchor_body_name)
            
            observation_dims = [round(dims) for dims in self.metadata["observation_dims"]] # list[float]
            self.num_obs = sum(observation_dims)
            
            self.num_actions = len(self.metadata["action_scale"])
            self.action_scale = np.array(self.metadata["action_scale"], dtype=np.float32)
            
            # init datas
            self.qj_obs = np.zeros(self.num_actions, dtype=np.float32)
            self.dqj_obs = np.zeros(self.num_actions, dtype=np.float32)
            self.obs = np.zeros(self.num_obs)
            self.action = np.zeros(self.num_actions)
            
            self.ref_joint_pos = np.zeros(self.num_actions, dtype=np.float32)
            self.ref_joint_vel = np.zeros(self.num_actions, dtype=np.float32)
            self.ref_anchor_ori_w = np.zeros(4, dtype=np.float32)
            
            print("kp_lab: ", self.kps_lab)
            print("kd_lab: ", self.kds_lab)
            print("default_angles_lab: ", self.default_angles_lab)
            print("mj_joint_names: ", self.lab_joint_names)
            print("action_scale_lab: ", self.action_scale)
            print("mj2lab", self.mj2lab)
            
            print("GaeMimic policy initializing ...")
    
    def _load_motion(self):
        """
        Load motion data from the specified motion path.
        
        IMPORTANT: The start pose must be static, and end pose must be static as well.
        
        - joint_pos: (num_frames, num_joints) joint positions
        - joint_vel: (num_frames, num_joints) joint velocities
        - body_pos_w: (num_frames, num_bodies, 3) body positions
        - body_quat_w: (num_frames, num_bodies, 4) body quaternions
        - body_lin_vel_w: (num_frames, num_bodies, 3) body linear velocities
        - body_ang_vel_w: (num_frames, num_bodies, 3) body angular velocities
        """
        data = np.load(self.motion_path)
        self.motion_data = {
            "joint_pos": data["joint_pos"],
            "joint_vel": data["joint_vel"],
            "body_pos_w": data["body_pos_w"],
            "body_quat_w": data["body_quat_w"],
            "body_lin_vel_w": data["body_lin_vel_w"],
            "body_ang_vel_w": data["body_ang_vel_w"],
        }
        self.motion_length = self.motion_data["joint_pos"].shape[0]
        
    def _load_metadata(self):
        metadata = {}
        
        for prop in self.onnx_model.metadata_props:
            key = prop.key
            value = prop.value
            
            # Try to parse as list (comma-separated values)
            if "," in value:
                try:
                    # Try parsing as float list first
                    parsed_list = [float(x.strip()) for x in value.split(",")]
                    metadata[key] = parsed_list
                except ValueError:
                    # If parsing as float fails, keep as string list
                    metadata[key] = [x.strip() for x in value.split(",")]
            else:
                # Try to parse as single value
                try:
                    metadata[key] = float(value)
                except ValueError:
                    # Keep as string if not a number
                    metadata[key] = value
        
        return metadata

    def enter(self):

        self.counter_step = 0

        observation = {}
        for i, input_name in enumerate(self.input_name):
            observation[input_name] = np.zeros((1, self.ort_session.get_inputs()[i].shape[1]), dtype=np.float32)


        outputs_result = self.ort_session.run(None, observation)
        self.action = outputs_result[0]

        self.qj_obs = np.zeros(self.num_actions, dtype=np.float32)
        self.dqj_obs = np.zeros(self.num_actions, dtype=np.float32)
        self.obs = np.zeros(self.num_obs)
        
        self.action = np.zeros(self.num_actions)
        pass
       
    @staticmethod 
    def quat_mul(q1, q2):
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
        
    @staticmethod 
    def matrix_from_quat(q):
        w, x, y, z = q
        return np.array([
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)]
        ])

    @staticmethod 
    def yaw_quat(q):
        w, x, y, z = q
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
        return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    
    @ staticmethod
    def euler_single_axis_to_quat(angle, axis, degrees=False):
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
    
    @staticmethod
    def compute_projected_gravity(quat_w, gravity_w=None):
        """
        计算重力在局部坐标系(body frame)下的投影
        
        参数:
            quat_w: 四元数 [w, x, y, z] - body在世界坐标系的姿态
            gravity_w: 世界坐标系下的重力向量 [x, y, z]，默认为 [0, 0, -1] (单位向量)
        
        返回:
            projected_gravity_b: 重力在body局部坐标系的投影 [x, y, z]
        """
        if gravity_w is None:
            gravity_w = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        
        # 提取四元数分量
        w, x, y, z = quat_w
        
        # 构建旋转矩阵 (从世界坐标系到局部坐标系)
        # R = quat_to_rotation_matrix(q)
        R = np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ], dtype=np.float32)
        
        # 应用旋转矩阵的转置(逆变换): 将世界坐标系的重力向量转到局部坐标系
        projected_gravity_b = R.T @ gravity_w
        
        return projected_gravity_b

    def run(self):
        robot_quat = self.state_cmd.base_quat
        gravity_ori = self.compute_projected_gravity(robot_quat)
        
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
        
        self.ref_anchor_ori_w = self.motion_data["body_quat_w"][self.counter_step, self.motion_anchor_id]
        print(self.ref_anchor_ori_w)

        # 在第一帧提取当前机器人yaw方向，与参考动作yaw方向做差（与beyond mimic一致）
        if(self.counter_step < 2):
            init_to_anchor = self.matrix_from_quat(self.yaw_quat(self.ref_anchor_ori_w))
            world_to_anchor = self.matrix_from_quat(self.yaw_quat(robot_quat))
            self.init_to_world = world_to_anchor @ init_to_anchor.T
            print("self.init_to_world: ", self.init_to_world)
            self.counter_step += 1
            return

        motion_anchor_ori_b = self.matrix_from_quat(robot_quat).T @ self.init_to_world @ self.matrix_from_quat(self.ref_anchor_ori_w)
        

        ang_vel = self.state_cmd.ang_vel
        
        dqj = self.state_cmd.dq[self.mj2lab]
        
        # TODO ref_joint_pos
        self.ref_joint_pos = self.motion_data["joint_pos"][self.counter_step]
        self.ref_joint_vel = self.motion_data["joint_vel"][self.counter_step]
        
        # command motion_anchor_ori_b base_ang_vel projected_gravity joint_pos joint_vel previous_action
        observation = {}
        observation[self.input_name[0]] = np.concatenate((self.ref_joint_pos.reshape(1, -1), self.ref_joint_vel.reshape(1, -1)), axis=-1).astype(np.float32)
        observation[self.input_name[1]] = motion_anchor_ori_b[:,:2].reshape(1, -1).astype(np.float32)
        observation[self.input_name[2]] = ang_vel.reshape(1, -1).astype(np.float32)
        observation[self.input_name[3]] = gravity_ori.reshape(1, -1).astype(np.float32)
        observation[self.input_name[4]] = qj.reshape(1, -1).astype(np.float32)
        observation[self.input_name[5]] = dqj.reshape(1, -1).astype(np.float32)
        observation[self.input_name[6]] = self.action.reshape(1, -1).astype(np.float32)
        
        outputs_result = self.ort_session.run(None, observation)

        # 处理多个输出
        self.action = outputs_result[0]
        
        target_dof_pos_lab = self.action * self.action_scale + self.default_angles_lab
        target_dof_pos_mj = np.zeros(29)
        target_dof_pos_mj[self.mj2lab] = target_dof_pos_lab.squeeze(0)
        
        self.policy_output.actions = target_dof_pos_mj
        self.policy_output.kps[self.mj2lab] = self.kps_lab
        self.policy_output.kds[self.mj2lab] = self.kds_lab
        
        # update motion phase
        self.counter_step += 1
        
        if self.counter_step >= self.motion_length - 1:
            self.state_cmd.skill_cmd = FSMCommand.LOCO

    def exit(self):
        self.action.fill(0.0)
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
            return FSMStateName.SKILL_GAE_MIMIC