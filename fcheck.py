# import numpy as np
# import matplotlib.pyplot as plt

# def simulate_projectile_motion(initial_velocity, launch_angle):
#     """
#     Simulates and visualizes the motion of a projectile.

#     Parameters:
#         initial_velocity (float): Initial velocity of the projectile (m/s).
#         launch_angle (float): Launch angle in degrees.
#     """
#     # Constants
#     g = 9.81  # Acceleration due to gravity (m/s^2)

#     # Convert angle to radians
#     angle_rad = np.radians(launch_angle)

#     # Compute time of flight, maximum height, and range
#     time_of_flight = (2 * initial_velocity * np.sin(angle_rad)) / g
#     max_height = (initial_velocity**2 * np.sin(angle_rad)**2) / (2 * g)
#     range_of_projectile = (initial_velocity**2 * np.sin(2 * angle_rad)) / g

#     # Generate time points for the trajectory
#     t = np.linspace(0, time_of_flight, num=500)

#     # Calculate x and y coordinates
#     x = initial_velocity * np.cos(angle_rad) * t
#     y = initial_velocity * np.sin(angle_rad) * t - 0.5 * g * t**2

#     # Plot the trajectory
#     plt.figure(figsize=(10, 6))
#     plt.plot(x, y, label="Projectile Trajectory")
#     plt.title("Projectile Motion Simulation")
#     plt.xlabel("Distance (m)")
#     plt.ylabel("Height (m)")
#     plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
#     plt.grid()
#     plt.legend()
#     plt.show()

#     # Print results
#     print(f"Time of Flight: {time_of_flight:.2f} seconds")
#     print(f"Maximum Height: {max_height:.2f} meters")
#     print(f"Range: {range_of_projectile:.2f} meters")

# # Example usage
# if __name__ == "__main__":
#     print("--- Projectile Motion Simulator ---")
#     velocity = float(input("Enter the initial velocity (m/s): "))
#     angle = float(input("Enter the launch angle (degrees): "))
#     simulate_projectile_motion(velocity, angle)

import numpy as np
import matplotlib.pyplot as plt

def newton_method(h, h_prime, x0, tol=1e-6, max_iter=100):
    """
    Finds a root of the function h(x) = 0 using Newton's method.

    Parameters:
        h (function): The function for which the root is sought.
        h_prime (function): The derivative of h(x).
        x0 (float): Initial guess.
        tol (float): Tolerance for stopping.
        max_iter (int): Maximum number of iterations.

    Returns:
        float: The root of h(x) if found within the given tolerance.
    """
    x = x0
    for i in range(max_iter):
        h_x = h(x)
        h_prime_x = h_prime(x)
        if abs(h_x) < tol:
            print(f"Converged to root at x = {x:.6f} after {i} iterations.")
            return x
        if h_prime_x == 0:
            raise ValueError("Derivative is zero. Newton's method fails.")
        x = x - h_x / h_prime_x
    raise ValueError("Maximum iterations reached. No convergence.")

# Define the functions f(x) and g(x)
def f(x):
    return np.sin(x)

def g(x):
    return 0.5 * x

# Define h(x) = f(x) - g(x) and its derivative h'(x)
def h(x):
    return f(x) - g(x)

def h_prime(x):
    return np.cos(x) - 0.5

# Visualize the functions and their intersection points
x_vals = np.linspace(0, 10, 500)
y_f = f(x_vals)
y_g = g(x_vals)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_f, label="f(x) = sin(x)")
plt.plot(x_vals, y_g, label="g(x) = 0.5x")
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Intersection of f(x) and g(x)")
plt.grid()
plt.legend()

# Use Newton's method to find intersection points
initial_guesses = [2, 5, 8]  # Initial guesses near the expected intersections
roots = []
for guess in initial_guesses:
    root = newton_method(h, h_prime, guess)
    roots.append(root)
    plt.scatter(root, f(root), color='red', label=f"Intersection at x={root:.2f}")

plt.legend()
plt.show()

# Print the results
print("Intersection points:")
for root in roots:
    print(f"x = {root:.6f}, y = {f(root):.6f}")
