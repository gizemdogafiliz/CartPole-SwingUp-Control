# Cart-Pole Swing-Up Control

Advanced cart-pole control implementation combining differentiable simulation (JAX) with high-fidelity physics simulation (MuJoCo). Features multiple control strategies including neural networks and mixture-of-experts.

## 🎯 What It Does

The cart-pole system consists of a cart moving horizontally with a pole that swings freely. This project implements controllers that:

1. **Swing up** the pole from hanging position (θ = π) to upright (θ = 0)
2. **Stabilize** the system around the equilibrium point
3. **Keep the cart** centered near x = 0

## 🚀 Five Control Approaches

| Controller | Method | Purpose |
|------------|---------|---------|
| **Linear PD** | Gradient-optimized PD control | Baseline stabilization |
| **LQR** | Linear-quadratic regulator | Optimal linear control near equilibrium |
| **Neural Network** | MLP trained via differentiable simulation | Nonlinear energy-based swing-up |
| **Hybrid (NN+LQR)** | Sequential switching | NN for swing-up, LQR for stabilization |
| **Mixture-of-Experts** | Learned state-dependent weighting | Adaptive blending of NN and LQR |

## 🏗️ Project Structure
```
CartPole-SwingUp-Control/
├── controller/              # Control algorithms
├── env/                     # Cart-pole dynamics & closed-loop sim
├── lib/                     # Training utilities and plotting
├── mujoco_simulation/       # MuJoCo physics simulation
│   ├── cartpole_wheels.xml
│   ├── cartpole_system.py
│   └── mujoco_env.yml
├── examples/                # Main experiment scripts
│   ├── Q1_main_linear_only.py
│   ├── Q2_main_nn_only.py
│   ├── Q2_main_nn_LQR.py
│   └── Q3_main_moe.py
├── main.ipynb               # Interactive notebook
├── requirements.txt
└── me58006_project_env.yml
```

## 🛠️ Installation
```bash
# Create conda environment
conda env create -f me58006_project_env.yml
conda activate me58006_project_env

# For MuJoCo simulation (optional)
conda env create -f mujoco_simulation/mujoco_env.yml
conda activate mujoco_env
```

**Dependencies**: JAX, Equinox, Optax, Diffrax, MuJoCo, NumPy, Matplotlib

## 🎮 Quick Start

### Run Experiments
```bash
# Compare Linear PD and LQR
python examples/Q1_main_linear_only.py

# Neural network swing-up
python examples/Q2_main_nn_only.py

# Hybrid approach (NN + LQR)
python examples/Q2_main_nn_LQR.py

# Mixture-of-Experts
python examples/Q3_main_moe.py

# Interactive notebook
jupyter notebook main.ipynb
```

### MuJoCo Visualization
```bash
cd mujoco_simulation
python cartpole_system.py
```

## 🔬 How It Works

### 1. Differentiable Simulation
- **JAX** enables automatic differentiation through entire simulation trajectories
- **Diffrax** provides differentiable ODE solvers for cart-pole dynamics integration
- Controllers optimized via gradient descent on trajectory costs

### 2. Neural Network Training
- **Input**: 5D state `[x, cos(θ), sin(θ), ẋ, θ̇]`
- **Output**: Control force `u`
- **Architecture**: MLP (5 → 64 → 64 → 1)
- **Loss**: Energy-based + position penalty
- **Optimizer**: Adam via Optax (lr=1e-3, 1000 epochs)

### 3. MuJoCo Deployment
<p align="center">
  <img src="mujoco_simulation/mujoco_sim.png" width="700"/>
  <br>
  <i>Interactive MuJoCo visualization of the cart-pole system</i>
</p>
- High-fidelity physics simulation with realistic dynamics
- Real-time 3D visualization with interactive viewer

## 📊 Key Features

- **JIT Compilation** - Fast execution with JAX
- **Multiple Control Strategies** - Compare classical and learning-based methods
- **Hybrid Controllers** - Combine strengths of different approaches
- **Comprehensive Analysis** - Trajectory costs, control efforts, state evolution plots
- **Modular Design** - Easy to extend with new controllers or cost functions

---

**Course**: ME 58006 - Deep Learning for Robot Control  
**Institution**: Sabancı University  
**Instructor**: Asst. Prof. Aykut Cihan Satıcı

**Built with**: JAX, MuJoCo, Equinox, Optax, Diffrax

