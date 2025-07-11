import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--show', action='store_true', default=True)
parser.add_argument('--interval', type=int, default=int(1000 / 60))  # ~60 FPS
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === GNN Model ===
class CollisionAvoidanceGNN(torch.nn.Module):
    def __init__(self, in_channels=6, hidden_channels=32):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, 3)  # Predict velocity delta (dx, dy, dz)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        dx = self.conv3(x, edge_index)
        return dx

# === Build graph and features ===
def build_graph(positions, goal, threshold=50.0):
    N = positions.shape[0]
    edges = []
    for i in range(N):
        for j in range(i+1, N):
            if np.linalg.norm(positions[i] - positions[j]) < threshold:
                edges.append([i, j])
                edges.append([j, i])
    if len(edges) == 0:
        edges = [[i, i] for i in range(N)]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)

    # Normalize positions
    pos_mean = positions.mean(axis=0)
    pos_std = positions.std(axis=0) + 1e-8
    pos_norm = (positions - pos_mean) / pos_std

    # Direction to goal normalized per node
    dir_to_goal = goal.reshape(1, 3) - positions
    dist_to_goal = np.linalg.norm(dir_to_goal, axis=1, keepdims=True) + 1e-8
    dir_norm = dir_to_goal / dist_to_goal

    # Node features: [pos_x, pos_y, pos_z, dir_x, dir_y, dir_z]
    features = np.concatenate([pos_norm, dir_norm], axis=1)
    x = torch.tensor(features, dtype=torch.float).to(device)

    return Data(x=x, edge_index=edge_index), pos_mean, pos_std

# === Environment ===
class DummyEnv:
    def __init__(self, num_agents=33):
        self.num_agents = num_agents
        self.goal = np.zeros(3)
        self.reset()

    def reset(self):
        self.positions = np.random.uniform(-100, 100, (self.num_agents, 3))
        self.velocities = np.zeros_like(self.positions)
        self.set_random_goal()

    def kill_first_node(self):
        if self.positions.shape[0] > 0:
            self.positions = np.delete(self.positions, 0, axis=0)
            self.velocities = np.delete(self.velocities, 0, axis=0)
            self.num_agents -= 1

    def set_random_goal(self):
        self.goal = np.random.uniform(-100, 100, 3)
        self.velocities[:] = 0

    def step(self, train=True):
        global current_loss

        graph, pos_mean, pos_std = build_graph(self.positions, self.goal)
        x, edge_index = graph.x, graph.edge_index

        if train:
            optimizer.zero_grad()
            dx = gnn_model(x, edge_index)  # velocity delta in normalized space

            next_pos_norm = x[:, :3] + dx * 0.1  # apply small step in normalized space
            dx_real = dx.detach().cpu().numpy() * pos_std  # scale to real space velocity delta

            next_pos_real = self.positions + dx_real * 0.1

            # Goal attraction loss (in real space)
            goal_vec = self.goal.reshape(1,3) - next_pos_real
            goal_dist = np.linalg.norm(goal_vec, axis=1)
            loss_goal = torch.mean(torch.tensor(goal_dist, device=device, dtype=torch.float))

            # Collision avoidance loss (in normalized space)
            dists = torch.cdist(next_pos_norm, next_pos_norm)
            mask = ~torch.eye(dists.size(0), dtype=torch.bool, device=dists.device)
            too_close = (dists < 2.0) & mask
            loss_collision = ((2.0 - dists[too_close]) ** 2).sum()

            # Velocity magnitude regularization
            loss_vel = torch.norm(dx, dim=1).mean()

            # Total loss
            loss = 5.0 * loss_collision + 1.0 * loss_goal + 0.1 * loss_vel
            loss.backward()
            optimizer.step()
            current_loss = loss.item()
        else:
            with torch.no_grad():
                dx = gnn_model(x, edge_index)
                dx_real = dx.cpu().numpy() * pos_std
                current_loss = None

        # Update velocities and positions in real space
        dx_real = dx.detach().cpu().numpy() * pos_std
        self.velocities = self.velocities * 0.7 + dx_real * 0.3
        self.velocities = np.clip(self.velocities, -2.0, 2.0)
        self.positions += self.velocities

# === Setup ===
gnn_model = CollisionAvoidanceGNN(in_channels=6).to(device)
optimizer = torch.optim.Adam(gnn_model.parameters(), lr=1e-3)

env = DummyEnv()
paused = False
training = False
current_loss = None

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# === Keyboard Controls ===
def interact(event):
    global paused, training
    if event.key == ' ':
        paused = not paused
        print("Paused" if paused else "Running")
    elif event.key == 'r':
        env.reset()
        print("Reset environment")
    elif event.key == 'k':
        env.kill_first_node()
        print("Killed first node")
    elif event.key == 't':
        training = not training
        print("Training ON" if training else "Training OFF")
    elif event.key == 'g':
        env.set_random_goal()
        print("New Goal:", env.goal)

# === Utility: Dynamic Axis limits ===
def update_axes_limits_dynamic(ax, arrays, margin=10.0):
    all_points = np.concatenate(arrays, axis=0)
    mins = all_points.min(axis=0) - margin
    maxs = all_points.max(axis=0) + margin
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])

# === Main ===
def main():
    if args.show:
        scat = ax.scatter(env.positions[:, 0], env.positions[:, 1], env.positions[:, 2], c='blue', s=100)
        loss_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)
        info_text = ax.text2D(0.02, 0.05,
                             "Keys:\n  [Space] pause\n  [t] toggle train\n  [g] new goal\n  [r] reset\n  [k] kill",
                             transform=ax.transAxes)

        update_axes_limits_dynamic(ax, [env.positions, env.goal.reshape(1, 3)])

        state = {
            'scat': scat,
            'quivers': [],
            'goal_scat': ax.scatter(env.goal[0], env.goal[1], env.goal[2], c='green', s=200, marker='*')
        }

        def animate(frame):
            global current_loss
            if not paused:
                env.step(train=training)

                pos = env.positions
                state['scat'].remove()
                state['scat'] = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c='blue', s=100)

                # Update goal marker
                state['goal_scat']._offsets3d = ([env.goal[0]], [env.goal[1]], [env.goal[2]])

                # Clear previous velocity arrows
                for q in state['quivers']:
                    q.remove()
                state['quivers'].clear()

                # Draw velocity arrows
                for i in range(len(pos)):
                    x, y, z = pos[i]
                    u, v, w = env.velocities[i]
                    q = ax.quiver(x, y, z, u, v, w, color='red', length=5.0, normalize=True)
                    state['quivers'].append(q)

                update_axes_limits_dynamic(ax, [pos, env.goal.reshape(1, 3)])

                if current_loss is not None:
                    loss_text.set_text(f"Loss: {current_loss:.4f}")

        fig.canvas.mpl_connect('key_press_event', interact)
        anim = FuncAnimation(fig, animate, interval=args.interval, cache_frame_data=False)
        plt.show()
    else:
        for i in range(10000):
            env.step(train=True)
            if i % 100 == 0 and current_loss is not None:
                print(f"Step {i}, Loss: {current_loss:.4f}")

if __name__ == '__main__':
    main()
