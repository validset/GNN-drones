import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--show', action='store_true', default=True)
parser.add_argument('--interval', type=int, default=int(1000 / 60))
args = parser.parse_args()

button_rng = [-330, 330, 3]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ConvergeToTargetGNN(torch.nn.Module):
    def __init__(self, in_channels=9, hidden_channels=64):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, 3)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        dx = self.conv3(x, edge_index)
        # dx = torch.tanh(dx)
        return dx

gnn_model = ConvergeToTargetGNN(in_channels=9).to(device)
optimizer = torch.optim.Adam(gnn_model.parameters(), lr=5e-4)

def build_graph(positions, velocities, target, threshold=55.0):
    N = positions.shape[0]
    rel_to_target = positions - target.reshape(1, 3)
    features = np.hstack([positions, rel_to_target, velocities])
    features = np.clip(features, -10, 10)

    edge_index = []
    for i in range(N):
        for j in range(N):
            if i != j and np.linalg.norm(positions[i] - positions[j]) < threshold:
                edge_index.append([i, j])

    if len(edge_index) == 0:
        edge_index = [[i, i] for i in range(N)]

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    x = torch.tensor(features, dtype=torch.float)
    return Data(x=x, edge_index=edge_index)

class DummyEnv:
    def __init__(self, num_agents=33):
        self.s_agents = num_agents
        self.reset()

    def reset(self):
        self.positions = np.random.uniform(-110, 110, (self.s_agents, 3))
        self.velocities = np.zeros((self.s_agents, 3))
        self.target = np.random.uniform(*button_rng)
        self.distances = None

    def set_target(self, new_target):
        print(f"New target set: {new_target}")
        self.target = new_target

    def kill_first_node(self):
        ax.set_xscale('linear')
        ax.set_yscale('linear')
        ax.set_zscale('linear')
        if self.positions.shape[0] > 1:
            self.positions = np.delete(self.positions, 0, axis=0)
            self.velocities = np.delete(self.velocities, 0, axis=0)
        else:
            print("Last agent cannot be killed.")

    def step(self, train=True):
        global current_loss
        if self.positions.shape[0] == 0:
            current_loss = None
            return

        graph = build_graph(self.positions, self.velocities, self.target)
        x = graph.x.to(device)
        edge_index = graph.edge_index.to(device)
        N = self.positions.shape[0]

        pos_tensor = torch.tensor(self.positions, dtype=torch.float, device=device)
        target_tensor = torch.tensor(self.target, dtype=torch.float, device=device).unsqueeze(0)

        if train:
            optimizer.zero_grad()
            dx = gnn_model(x, edge_index)
            dx_drones = dx[:N]
            next_positions = pos_tensor + dx_drones
            # loss = F.mse_loss(next_positions, target_tensor.expand_as(next_positions))
            loss = F.mse_loss(next_positions, target_tensor.expand_as(next_positions))
            loss.backward()
            optimizer.step()
            current_loss = loss.item()
            dx_drones = dx_drones.detach().cpu().numpy()
        else:
            with torch.no_grad():
                dx = gnn_model(x, edge_index)
                dx_drones = dx[:N].cpu().numpy()
                current_loss = None

        self.positions += dx_drones
        pos_tensor = torch.tensor(self.positions, dtype=torch.float)
        self.distances = torch.cdist(pos_tensor, pos_tensor)

paused = False
current_loss = None
training_enabled = False

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

controls_text = fig.text(
    0.01, 0.01,
    "Controls:\n[space] Pause/Resume\n[r] Reset\n[k] Kill first agent\n[t] Toggle training\n[g] New goal",
    fontsize=10,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5)
)

info_box = fig.text(
    0.01, 0.91,
    "",  # filled in animation
    fontsize=10,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5)
)

env = DummyEnv()

def interact(event):
    global paused, training_enabled
    if event.key == ' ':
        paused = not paused
        print('Paused' if paused else 'Running')
    elif event.key == 'r':
        print("Reset")
        env.reset()
    elif event.key == 'k':
        print("Kill first agent")
        env.kill_first_node()
    elif event.key == 't':
        training_enabled = not training_enabled
        print("Training ON" if training_enabled else "Training OFF")
    elif event.key == 'g':
        new_target = np.random.uniform(*button_rng)
        env.set_target(new_target)
        print(f"New target generated: {new_target}")

def update_axes_limits_dynamic(ax, positions, target, margin=5.0):
    if positions.shape[0] == 0:
        return
    combined = np.vstack([positions, target.reshape(1, 3)])
    mins = np.min(combined, axis=0) - margin
    maxs = np.max(combined, axis=0) + margin
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])

def main():
    if args.show:
        scat_agents = ax.scatter(env.positions[:, 0], env.positions[:, 1], env.positions[:, 2], c='blue', s=100)
        scat_target = ax.scatter([env.target[0]], [env.target[1]], [env.target[2]], c='red', s=250, marker='X')

        frame_counter = 0

        def animate(frame):
            nonlocal scat_agents, scat_target, frame_counter

            if not paused:
                env.step(train=training_enabled)
                positions = env.positions

                if positions.shape[0] == 0:
                    if scat_agents is not None:
                        scat_agents.remove()
                        scat_agents = None
                    info_box.set_text("No agents left")
                    return

                colors = ['blue'] * positions.shape[0]
                if env.distances is not None:
                    close_pairs = (env.distances < 2.0) & ~torch.eye(env.distances.size(0), dtype=torch.bool)
                    for i in range(positions.shape[0]):
                        if torch.any(close_pairs[i]):
                            min_dist = env.distances[i][close_pairs[i]].min().item()
                            if min_dist < 1.0:
                                colors[i] = 'red'
                            else:
                                colors[i] = 'yellow'

                scat_agents._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
                scat_agents.set_color(colors)
                scat_target._offsets3d = ([env.target[0]], [env.target[1]], [env.target[2]])

                update_axes_limits_dynamic(ax, positions, env.target)

            mode = "Training" if training_enabled else "Inference"
            target_str = f"[{env.target[0]:.1f}, {env.target[1]:.1f}, {env.target[2]:.1f}]"
            if current_loss is not None:
                info_box.set_text(
                    f"Loss: {current_loss:.4f}\nTarget: {target_str}\nMode: {mode}"
                )
                if frame_counter % 100 == 0:
                    print(f"Frame {frame}: Loss = {current_loss:.4f}")
            else:
                info_box.set_text(
                    f"Loss: ---\nTarget: {target_str}\nMode: {mode}"
                )

            frame_counter += 1

        fig.canvas.mpl_connect('key_press_event', interact)
        ani = FuncAnimation(fig, animate, interval=args.interval, cache_frame_data=False)
        plt.show()

if __name__ == '__main__':
    main()
