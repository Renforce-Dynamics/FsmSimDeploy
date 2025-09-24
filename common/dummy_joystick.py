import pygame
from pygame.locals import *
from enum import IntEnum, unique
import threading
import time

@unique
class JoystickButton(IntEnum):
    # Standard PlayStation/Xbox Layout
    A = 0      # PS: Cross(×), Xbox: A
    B = 1      # PS: Circle(○), Xbox: B
    X = 2      # PS: Square(□), Xbox: X
    Y = 3      # PS: Triangle(△), Xbox: Y
    L1 = 4     # Left Bumper (L1 on PS)
    R1 = 5     # Right Bumper (R1 on PS)
    SELECT = 6   # Select/Share button
    START = 7  # Start/Options button
    L3 = 8     # Left Stick Press
    R3 = 9     # Right Stick Press
    HOME = 10  # PS: PS FSMCommand, Xbox: Xbox FSMCommand
    UP = 11    # D-pad Up (if mapped as separate button)
    DOWN = 12  # D-pad Down
    LEFT = 13  # D-pad Left
    RIGHT = 14 # D-pad Right

class DummyJoyStick:
    """
    Keyboard Joystick Controller Simulator
    
    Key Mappings:
    - WASD: Left stick control (W/S: forward/backward, A/D: left/right)
    - Arrow keys: Right stick control
    - Space: A button (Skill 1)
    - X: X button (Skill 2) 
    - C: Y button (Skill 3)
    - V: B button (Skill 4)
    - Q: L1 button
    - E: R1 button
    - Tab: SELECT button (Exit)
    - Enter: START button (Position Reset)
    - Z: L3 button (Passive Mode)
    - R: R3 button (Motion Mode)
    """
    
    def __init__(self):
        pygame.init()
        
        # Create a small window to capture keyboard events
        self.screen = pygame.display.set_mode((400, 300))
        pygame.display.set_caption("Keyboard Joystick Controller")
        
        # Button states
        self.button_count = 15
        self.button_states = [False] * self.button_count
        self.button_released = [False] * self.button_count
        self.prev_button_states = [False] * self.button_count
        
        # Axis states (simulated joysticks)
        self.axis_count = 4
        self.axis_states = [0.0] * self.axis_count  # [LeftX, LeftY, RightX, RightY]
        
        # Hat switch states (D-pad)
        self.hat_count = 1
        self.hat_states = [(0, 0)]
        
        # Keyboard key mappings
        self.key_mappings = {
            # Left stick (WASD)
            pygame.K_w: ('left_stick_y', -1.0),  # Forward
            pygame.K_s: ('left_stick_y', 1.0),   # Backward
            pygame.K_a: ('left_stick_x', -1.0),  # Left
            pygame.K_d: ('left_stick_x', 1.0),   # Right
            
            # Right stick (Arrow keys)
            pygame.K_UP: ('right_stick_y', -1.0),
            pygame.K_DOWN: ('right_stick_y', 1.0),
            pygame.K_LEFT: ('right_stick_x', -1.0),
            pygame.K_RIGHT: ('right_stick_x', 1.0),
            
            # Buttons
            pygame.K_j: JoystickButton.A,          # A button
            pygame.K_h: JoystickButton.X,          # X button
            pygame.K_u: JoystickButton.Y,          # Y button
            pygame.K_k: JoystickButton.B,          # B button
            pygame.K_y: JoystickButton.L1,         # L1
            pygame.K_i: JoystickButton.R1,         # R1
            pygame.K_TAB: JoystickButton.SELECT,   # SELECT
            pygame.K_RETURN: JoystickButton.START, # START
            pygame.K_t: JoystickButton.L3,         # L3
            pygame.K_o: JoystickButton.R3,         # R3 (Motion mode)
        }
        
        # Currently pressed keys
        self.pressed_keys = set()
        
        print("Keyboard Joystick Controller Started!")
        print("Control Instructions:")
        print("  WASD: Left stick (Movement)")
        print("  Arrow keys: Right stick (Rotation)")
        print("  Space: A button (Use with R1 - Motion mode)")
        print("  X: X button (Use with R1 - Skill 1)")
        print("  C: Y button (Use with R1 - Skill 2, Use with L1 - Skill 4)")
        print("  V: B button (Use with R1 - Skill 3)")
        print("  Q: L1 button")
        print("  E: R1 button")
        print("  Tab: SELECT (Exit)")
        print("  Enter: START (Position Reset)")
        print("  Z: L3 (Passive mode)")
        print("  R: R3 (Motion mode)")
    
    def update(self):
        """Update joystick state"""
        # Save previous frame button states
        self.prev_button_states = self.button_states.copy()
        
        # Reset release states
        self.button_released = [False] * self.button_count
        
        # Reset axis states
        self.axis_states = [0.0] * self.axis_count
        
        # Reset button states
        self.button_states = [False] * self.button_count
        
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                self.pressed_keys.add(event.key)
            elif event.type == pygame.KEYUP:
                self.pressed_keys.discard(event.key)
        
        # Process currently pressed keys
        for key in self.pressed_keys:
            if key in self.key_mappings:
                mapping = self.key_mappings[key]
                
                if isinstance(mapping, tuple):
                    # Axis control
                    stick_type, value = mapping
                    if stick_type == 'left_stick_x':
                        self.axis_states[0] += value
                    elif stick_type == 'left_stick_y':
                        self.axis_states[1] += value
                    elif stick_type == 'right_stick_x':
                        self.axis_states[2] += value
                    elif stick_type == 'right_stick_y':
                        self.axis_states[3] += value
                else:
                    # Button control
                    button_id = mapping
                    if 0 <= button_id < self.button_count:
                        self.button_states[button_id] = True
        
        # Limit axis values to [-1.0, 1.0] range
        for i in range(self.axis_count):
            self.axis_states[i] = max(-1.0, min(1.0, self.axis_states[i]))
        
        # Detect button release events
        for i in range(self.button_count):
            if self.prev_button_states[i] and not self.button_states[i]:
                self.button_released[i] = True
        
        # Update display
        self.screen.fill((0, 0, 0))
        font = pygame.font.Font(None, 24)
        
        # Display current status
        y_offset = 20
        text_lines = [
            "Keyboard Joystick Controller",
            f"Left Stick: ({self.axis_states[0]:.2f}, {self.axis_states[1]:.2f})",
            f"Right Stick: ({self.axis_states[2]:.2f}, {self.axis_states[3]:.2f})",
            f"Button Status: {[i for i, pressed in enumerate(self.button_states) if pressed]}",
            "",
            "Press Tab to Exit"
        ]
        
        for line in text_lines:
            text = font.render(line, True, (255, 255, 255))
            self.screen.blit(text, (10, y_offset))
            y_offset += 30
        
        pygame.display.flip()
    
    def is_button_pressed(self, button_id):
        """Check if button is pressed"""
        if 0 <= button_id < self.button_count:
            return self.button_states[button_id]
        return False
    
    def is_button_released(self, button_id):
        """Check if button is released"""
        if 0 <= button_id < self.button_count:
            return self.button_released[button_id]
        return False
    
    def get_axis_value(self, axis_id):
        """Get axis value"""
        if 0 <= axis_id < self.axis_count:
            return self.axis_states[axis_id]
        return 0.0
    
    def get_hat_direction(self, hat_id=0):
        """Get hat switch direction"""
        if 0 <= hat_id < self.hat_count:
            return self.hat_states[hat_id]
        return (0, 0)
    
    def cleanup(self):
        """Clean up resources"""
        pygame.quit()
