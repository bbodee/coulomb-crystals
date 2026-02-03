"""
Potential functions and force calculations for 1D Coulomb Crystal Simulation
"""

import numpy as np

class Potential:
    """Base class for potential functions"""

    def __call__(self, x):
        """Calculate potential energy at position x"""
        raise NotImplementedError
    
    def force(self, x):
        """Calculate force at position x (f = -dV/dx)"""
        raise NotImplementedError
    
class NoPotential(Potential):
    """Free particle - no confining potential"""

    def __call__(self, x):
        return 0.0
    
    def force(self, x):
        return 0.0
    
class HarmonicPotential(Potential):
    """Harmonic oscillator potential: V(x) = 0.5* k *x^2"""

    def __init__(self, k=1.0, center=0.0):
        """
        Parameters:
        -----------
        k : float
            Spring constant (dimensionless)
        center : float
            Center position of the potential
        """

        self.k = k
        self.center = center

    def __call__(self, x):
        """Potential Energy"""

        dx = x- self.center
        return 0.5 * self.k * dx**2
    
    def force(self, x):
        """Force: F = -k * (x-center)"""
        return - self.k * (x - self.center)
    
class QuarticPotential(Potential):
    """Quartic potential: V(x) = a * x^4"""

    def __init__(self, a=1.0, center=0.0):
        """
        Parameters:
        -----------
        a : float
            Quartic coefficient (dimensionless)
        
        center : float
            Center position of the potential
        """

        self.a = a
        self.center = center

    def __call__(self, x):
        """Potential energy"""
        dx = x - self.center
        return self.a * dx**4
    
    def force(self, x):
        """Force: F = -4 * a * x^3"""
        dx = x - self.center
        return -4.0 * self.a * dx**3
    
class BoxPotential(Potential):
    """Infinite Square Well Potential"""

    def __init__(self, width=2.0, center=0.0, wall_stiffness=1000.0):
        """
        Parameters:
        -----------
        width : float
            Width of the box
        center : float
            Center of the box
        wall_stiffness : float
            Stiffness of the walls (for soft walls)
        """
        self.width = width
        self.center = center
        self.wall_stiffness = wall_stiffness
        self.half_width = width / 2.0

    def __call__(self, x):
        """Potential energy (infinite outsidem zero inside)"""
        dx = abs(x - self.center)
        if dx <= self.half_width:
            return 0.0
        else:
            # Soft wall approximation
            overshoot = dx - self.half_width
            return 0.5 * self.wall_stiffness * overshoot**2
        
    def force(self, x):
        """Force from walls"""
        dx = x - self.center
        if abs(dx) <= self.half_width:
            return 0.0
        else:
            # Soft wall force
            sign = np.sign(dx)
            overshoot = abs(dx) - self.half_width
            return - sign * self.wall_stiffness * overshoot
