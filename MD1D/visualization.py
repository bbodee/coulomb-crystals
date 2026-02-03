"""
Visualization functions for 1D Coulomb Crystal simulation
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def plot_trajectory(times, positions, velocities, energies=None, particle_info=None):
    """
    Plot particle trajectory, velocity, and enerygy

    Parameters:
    -----------
    times : array
        Time points
    positions : array
        Position history
    velocities : array
        Velocity history
    energies : dict, optional
        Dictioanry with 'kinetic', 'potential', 'total' energy arrays
    particle_info : str, optional
        Additional particle information for title
    """

    n_plots = 3 if energies is not None else 2
    fig, axes = plt.subplots(n_plots, 1, figsize=(10,4*n_plots))

    # Position vs time plot
    axes[0].plot(times, positions, 'b-', linewidth=2)
    axes[0].set_xlabel('Time (dimensionless)', fontsize=12)
    axes[0].set_ylabel('Position (dimensionless)', fontsize=12)
    title = 'Particle Motion'
    if particle_info:
        title += f': {particle_info}'
    axes[0].set_title(title, fontsize=14)
    axes[0].grid(True, alpha=0.3)

    # Velocity vs time
    axes[1].plot(times, velocities, 'r-', linewidth=2)
    axes[1].set_xlabel('Time (dimensionless)', fontsize=12)
    axes[1].set_ylabel('Velocity (dimensionless)', fontsize=12)
    axes[1].set_title('Velocity vs Time', fontsize=14)
    axes[1].grid(True, alpha=0.3)

    # Energy vs time
    if energies is not None:
        axes[2].plot(times, energies['kinetic'], 'g-', linewidth=2, label='Kinetic')
        axes[2].plot(times, energies['potential'], 'orange', linewidth=2, label='Potential')
        axes[2].plot(times, energies['total'], 'k--', linewidth=2, label='Total')
        axes[2].set_xlabel('Time (dimensionless)', fontsize=12)
        axes[2].set_ylabel('Energy (dimensionless)', fontsize=12)
        axes[2].set_title('Energy vs Time', fontsize=14)
        axes[2].legend(fontsize=10)
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_phase_space(positions, velocities):
    """Plot phase space (position vs velocity)"""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Color by time
    colors = np.arange(len(positions))
    scatter = ax.scatter(positions, velocities, c=colors, cmap='viridis',
                         alpha=0.6, s=20)
    
    ax.set_xlabel('Position (dimensionless)', fontsize=12)
    ax.set_ylabel('Velocity (dimensionless)', fontsize=12)
    ax.set_title('Phase Space Trajectory', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Add colorbar to show time progression
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Time', fontsize=12)

    plt.tight_layout()
    plt.show()

def animate_particles(times, positions, potential=None):
    """
    Create an animation of the particle moving in 1D space.

    Parameters:
    -----------
    times : array
        Time points
    positions : array
        Position history
    potential : Potential, optional
        Potential function to visualize
    """

    fig, (ax1, ax2) = plt.subplots(2,1,figsize=(12,8))

    # Determine plot range
    pos_min, pos_max = positions.min(), positions.max()
    margin = (pos_max - pos_min) * 0.2
    x_min, x_max = pos_min - margin, pos_max + margin

    # Top plot: Particle Motion
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(-0.5, 0.5)
    ax1.set_xlabel('Position (Dimensionless)', fontsize=12)
    ax1.set_title('1D Particle Motion', fontsize=14)
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    particle_dot, = ax1.plot([] , [], 'ro', markersize=15)
    trail_line, = ax1.plot([], [], 'b-', alpha=0.3, linewidth=1)
    time_text= ax1.text(0.02, 0.95, '', transform=ax1.transAxes, fontsize=12, verticalalignment='top')

    # Bottom plot: Potential
    ax2.set_xlim(x_min, x_max)
    ax2.set_xlabel('Position (dimensionless)', fontsize=12)
    ax2.set_ylabel('Potential Energy', fontsize=12)
    ax2.set_title('Confining Potential', fontsize=14)
    ax2.grid(True, alpha=0.3)

    if potential is not None:
        x_plot = np.linspace(x_min, x_max, 200)
        v_plot = np.array([potential(x) for x in x_plot])
        ax2.plot(x_plot, v_plot, 'k-', linewidth=2)
        ax2.set_ylim(v_plot.min() - 0.1, v_plot.max() + 0.1)

    particle_potential, = ax2.plot([], [], 'ro', markersize=10)

    def init():
        particle_dot.set_data([], [])
        trail_line.set_data([], [])
        particle_potential.set_data([], [])
        time_text.set_text('')
        return particle_dot, trail_line, particle_potential, time_text
    
    def update(frame):
        # Update particle position
        particle_dot.set_data([positions[frame]], [0])

        # Update trail (last 100 points)
        trail_start = max(0, frame - 100)
        trail_line.set_data(positions[trail_start: frame+1], np.zeros(frame - trail_start + 1))

        # Update particle on potential plot
        if potential is not None:
            particle_potential.set_data([positions[frame]], [potential(positions[frame])])

        # Update time
        time_text.set_text(f'Time: {times[frame]:.2f}')

        return particle_dot, trail_line, particle_potential, time_text
    
    # Create animation (sample every 5 frames for speed)
    anim = FuncAnimation(fig, update, init_func=init, frames=range(0, len(times), 5), interval=20, blit=True)

    plt.show()
    return anim

