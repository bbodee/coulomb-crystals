"""
1D Coulomb Crystal Simulation - Main simulation runner
Single particle with confining potential
"""

import numpy as np
from particle import Particle1D
from potentials import NoPotential, HarmonicPotential, QuarticPotential, BoxPotential
from visualization import plot_trajectory, plot_phase_space, animate_particles

def simulate_particle(particle, potential, duration=10.0, dt=0.01):
    """
    Run simulation for a single particle in a potentail

    Parameters:
    -----------
    particle : Particle1D
        The particle to simulate
    potential : Potential
        The confining potential
    duration : float
        Total simulation time
    dt : float
        Time step size
    
    Returns:
    --------
    times : array
        Time points
    positions :  array
        Position history
    velocities : array
        Velocity history
    energies :  dict
        Dictionary with kinetic, potential, and total energy arrays
    """

    time = 0.0
    times = [time]
    positions = [particle.position]
    velocities = [particle.velocity]

    # Energy tracking
    ke = [0.5 * particle.mass * particle.velocity**2]
    pe = [potential(particle.position)]
    total_e = [ke[0] + pe[0]]

    while time < duration:
        # Calculate force from potential
        force = potential.force(particle.position)

        # Update particle (Velocity Verlet integration)
        particle.apply_force(force)
        particle.update_velocity(dt / 2)
        particle.update_position(dt)

        # Recalculate force at new position
        force = potential.force(particle.position)
        particle.apply_force(force)
        particle.update_velocity(dt / 2) # second half step

        # Update time and record state
        time += dt
        times.append(time)
        positions.append(particle.position)
        velocities.append(particle.velocity)

        # Calculate energies
        ke.append(0.5 * particle.mass * particle.velocity**2)
        pe.append(potential(particle.position))
        total_e.append(ke[-1] + pe[-1])

    energies = {
        'kinetic': np.array(ke),
        'potential': np.array(pe),
        'total': np.array(total_e)
    }

    return np.array(times), np.array(positions), np.array(velocities), energies

def main():
    print("=" * 60)
    print("1D COULOMB CRYSTAL SIMULATION")
    print("Single Particle in Confining Potential")
    print("=" * 60)

    choice = input("\n Select potential (1-4) [default:2]: ").strip()
    if not choice:
        choice = '2'

    # Create potential based on choice
    if choice == '1':
        potential = NoPotential()
        pot_name = "No Potential"

    elif choice == '2':
        k = float(input("Enter spring constant k [default: 1.0]: ") or 1.0)
        potential = HarmonicPotential(k=k)
        pot_name = f'Harmonic (k={k})'
    elif choice == '3':
        a = float(input("Enter quartic coefficient a [default: 0.1]: ") or 2.0)
        potential = QuarticPotential(a=a)
        pot_name = f"Quartic (a={a})"
    elif choice == '4':
        width = float(input("Enter box width [default: 2.0]: ") or 2.0)
        potential = BoxPotential(width=width)
        pot_name = f"Box (width={width})"
    else:
        print("Invalid choice, using harmonic potential [2]")
        potential = HarmonicPotential()
        pot_name = "Harmonic (k=1.0)"

    print(f"\nUsing: {pot_name}")

    # Create particle
    print("\nParticle initialization:")
    use_random = input("Use random initial condtions? (y/n) [default: y]: ").strip().lower()
    if use_random == 'n':
        pos = float(input("enter initial position [default: 0.5]: ") or 0.5)
        vel = float(input("Enter initial velocity [default: 0.0]: ") or 0.0)
        particle = Particle1D(position=pos, velocity=vel)
    else:
        particle = Particle1D()

    print(f"\n Initial state: {particle}")

    # Simulation Parameters
    duration = float(input("\nSimulation duration [default: 20.0]: ") or 20.0)
    dt = float(input("Time step [default: 0.01]: ") or 0.01)

    # Run Simulation
    print("\nRunning Simulation...")
    times, positions, velocities, energies = simulate_particle(particle, potential, duration=duration, dt=dt)

    print(f"Smulaton complete!")
    print(f"Duration: {duration} (dimensionless time)")
    print(f"Time steps: {len(times)}")
    print(f"Final position: {positions[-1]:4f}")
    print(f"Final velocity: {velocities[-1]:4f}")
    print(f"\nEnergy conservation:")
    print(f"Initial total energy: {energies['total'][0]: .6f}")
    print(f"Final total energy: {energies['total'][-1]:.6f}")
    print(f"Energy drift: {abs(energies['total'][-1] - energies['total'][0]):.6e}")

    # Generate plots
    print("\nGenerating plots...")
    particle_info = f"pos0={positions[0]:.3f}, vel0={velocities[0]:.3f}, {pot_name}"
    plot_trajectory(times, positions, velocities, energies, particle_info)

    # Phase space plot
    show_phase = input("\nShow phase space plot? (y/n) [default: y]: ").strip().lower()
    if show_phase != 'n':
        plot_phase_space(positions, velocities)

    # Animation
    show_anim = input("\nShow animation? (y/n) [default:n]: ").strip().lower()
    if show_anim =='y':
        print("Generating animation...")
        animate_particles(times, positions, potential)


if __name__ == "__main__":
    main()
