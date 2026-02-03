"""
Particle Class for 1D Coulomb Crystal Simulation
"""

import numpy as np

class Particle1D:
    """A Single Particle in 1D dimensionless space"""

    def __init__(self, position=None, velocity=None, mass=1.0):
        """
        Initialize particle with random or specified position and velocity
        
        Parameters:
        -----------
        position : float, optional
            Initial position (dimensionless). If None, random in [0, 1]
        velocity : float, optional
            Initial velocity (dimensionless). If None, random in [-1, 1]
        mass : float, optional
            Particle mass (dimensionless). Default is 1.0
        """
        self.position = np.random.random() if position is None else position
        self.velocity = np.random.uniform(-1, 1) if velocity is None else velocity
        self.mass = mass
        self.acceleration = 0.0

    def apply_force(self, force):
        """
        Calculate acceleration from force using F = ma

        Parameters:
        -----------
        force : float
            Force acting on particle (dimensionless)
        """
        self.acceleration = force / self.mass

    def update_position(self, dt):
        """Update position based on velocity."""
        self.position += self.velocity * dt

    def update_velocity(self, dt):
        """Update velocity based on acceleration"""
        self.velocity += self.acceleration * dt

    def __repr__(self):
        return f"Particle(pos={self.position:.4f}, vel={self.velocity:.4f}, acc={self.acceleration:.4f})"
