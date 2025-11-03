import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from common.path_config import PROJECT_ROOT

import time
import mujoco.viewer
import mujoco
import numpy as np
import yaml
import os
from common.ctrlcomp import *
from FSM.FSM import *
from common.utils import get_gravity_orientation
import threading
import queue


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


class TerminalController:
    def __init__(self):
        self.command_queue = queue.Queue()
        self.running = True
        self.vel_cmd = np.zeros(3)
        self.input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self.input_thread.start()
        
    def _input_loop(self):
        """异步线程接收终端输入"""
        print("\n=== Terminal Controller ===")
        print("Commands:")
        print("  l3          - Passive mode")
        print("  start       - Position reset")
        print("  A+R1        - Locomotion mode")
        print("  X+R1        - Skill 1")
        print("  Y+R1        - Skill 2")
        print("  B+R1        - Skill 3")
        print("  Y+L1        - Skill 4")
        print("  B+L1        - Skill 5")
        print("  vel x y z   - Set velocity (e.g., 'vel 0.5 0 0.2')")
        print("  exit        - Exit program")
        print("===========================\n")
        
        while self.running:
            try:
                cmd = input("Enter command: ").strip().lower()
                if cmd:
                    self.command_queue.put(cmd)
            except EOFError:
                break
            except Exception as e:
                print(f"Input error: {e}")
                
    def get_command(self):
        """获取队列中的命令"""
        try:
            return self.command_queue.get_nowait()
        except queue.Empty:
            return None
            
    def cleanup(self):
        """清理资源"""
        self.running = False


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    mujoco_yaml_path = os.path.join(current_dir, "config", "mujoco.yaml")
    with open(mujoco_yaml_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        xml_path = os.path.join(PROJECT_ROOT, config["xml_path"])
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]
        
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    mj_per_step_duration = simulation_dt * control_decimation
    num_joints = m.nu
    policy_output_action = np.zeros(num_joints, dtype=np.float32)
    kps = np.zeros(num_joints, dtype=np.float32)
    kds = np.zeros(num_joints, dtype=np.float32)
    sim_counter = 0
    
    state_cmd = StateAndCmd(num_joints)
    policy_output = PolicyOutput(num_joints)
    FSM_controller = FSM(state_cmd, policy_output)
    
    # Use terminal controller
    controller = TerminalController()
    Running = True
    
    print("MuJoCo simulation started with terminal control!")
    
    with mujoco.viewer.launch_passive(m, d) as viewer:
        sim_start_time = time.time()
        while viewer.is_running() and Running:
            try:
                # Process terminal commands
                cmd = controller.get_command()
                if cmd:
                    if cmd == "exit":
                        print("Exit signal detected, shutting down...")
                        Running = False
                    elif cmd == "l3":
                        state_cmd.skill_cmd = FSMCommand.PASSIVE
                        print("Switched to passive mode")
                    elif cmd == "start":
                        state_cmd.skill_cmd = FSMCommand.POS_RESET
                        print("Position reset")
                    elif cmd == "a+r1":
                        state_cmd.skill_cmd = FSMCommand.LOCO
                        print("Motion mode")
                    elif cmd == "x+r1":
                        state_cmd.skill_cmd = FSMCommand.SKILL_1
                        print("Skill 1")
                    elif cmd == "y+r1":
                        state_cmd.skill_cmd = FSMCommand.SKILL_2
                        print("Skill 2")
                    elif cmd == "b+r1":
                        state_cmd.skill_cmd = FSMCommand.SKILL_3
                        print("Skill 3")
                    elif cmd == "y+l1":
                        state_cmd.skill_cmd = FSMCommand.SKILL_4
                        print("Skill 4")
                    elif cmd == "b+l1":
                        state_cmd.skill_cmd = FSMCommand.SKILL_5
                        print("Skill 5")
                    elif cmd.startswith("vel "):
                        try:
                            parts = cmd.split()
                            if len(parts) == 4:
                                state_cmd.vel_cmd[0] = float(parts[1])  # x
                                state_cmd.vel_cmd[1] = float(parts[2])  # y
                                state_cmd.vel_cmd[2] = float(parts[3])  # z
                                print(f"Velocity set to: {state_cmd.vel_cmd}")
                        except ValueError:
                            print("Invalid velocity format. Use: vel x y z")
                    else:
                        print(f"Unknown command: {cmd}")
                
                step_start = time.time()
                
                tau = pd_control(policy_output_action, d.qpos[7:], kps, np.zeros_like(kps), d.qvel[6:], kds)
                d.ctrl[:] = tau
                mujoco.mj_step(m, d)
                sim_counter += 1
                if sim_counter % control_decimation == 0:
                    
                    qj = d.qpos[7:]
                    dqj = d.qvel[6:]
                    quat = d.qpos[3:7]
                    
                    omega = d.qvel[3:6] 
                    gravity_orientation = get_gravity_orientation(quat)
                    
                    state_cmd.q = qj.copy()
                    state_cmd.dq = dqj.copy()
                    state_cmd.gravity_ori = gravity_orientation.copy()
                    state_cmd.base_quat = quat.copy()
                    state_cmd.ang_vel = omega.copy()
                    
                    FSM_controller.run()
                    policy_output_action = policy_output.actions.copy()
                    kps = policy_output.kps.copy()
                    kds = policy_output.kds.copy()
                    
            except ValueError as e:
                print(f"Error: {str(e)}")
            except Exception as e:
                print(f"Unexpected error: {str(e)}")
                break
            
            viewer.sync()
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    
    # Clean up resources
    controller.cleanup()
    print("Program exited")