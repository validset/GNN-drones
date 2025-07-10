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
parser.add_argument('--interval', type=int, default=int(1000 / 60))  # ~60 FPS
args = parser.parse_args()

# === GNN Definition ===
class CollisionAvoidanceGNN(torch.nn.Module):
    def __init__(self, in_channels=3, hidden_channels=32):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, in_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        dx = self.conv2(x, edge_index)
        return dx

gnn_model = CollisionAvoidanceGNN()
optimizer = torch.optim.Adam(gnn_model.parameters(), lr=1e-3)

def build_graph(positions, threshold=50.0):  # large threshold to connect most
    N = positions.shape[0]
    edge_index = []

    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(positions[i] - positions[j]) < threshold:
                edge_index.append([i, j])
                edge_index.append([j, i])

    if len(edge_index) == 0:
        edge_index = [[i, i] for i in range(N)]

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    x = torch.tensor(positions, dtype=torch.float)
    return Data(x=x, edge_index=edge_index)

class DummyEnv:
    def __init__(self, num_agents=101):
        self.n_agents = num_agents
        self.s_agents = num_agents
        self.positions = np.random.uniform(-101, 101, (num_agents, 3))
        self.velocities = np.random.randn(num_agents, 3) * 0.1
        self.food = np.random.uniform(-10, 10, (5, 3))  # static, unused here

    def reset(self):
        self.n_agents = self.s_agents
        self.positions = np.random.uniform(-101, 101, (self.s_agents, 3))  # reset to original count and random pos
        self.velocities = np.random.randn(self.s_agents, 3) * 0.1          # reset velocities
        self.food = np.random.uniform(-10, 10, self.food.shape)



    def kill_first_node(self):
        if len(self.positions) > 0:
            self.positions = np.delete(self.positions, 0, axis=0)
            self.velocities = np.delete(self.velocities, 0, axis=0)
            # hfood later


    def step(self, train=True):
        global current_loss
        graph = build_graph(self.positions)
        x = graph.x
        edge_index = graph.edge_index

        if train:
            optimizer.zero_grad()
            dx = gnn_model(x, edge_index)  # velocity changes prediction

            next_positions = x + dx * 0.1

            dists = torch.cdist(next_positions, next_positions)
            mask = ~torch.eye(dists.size(0), dtype=torch.bool, device=dists.device)

            # Penalty for agents closer than 2 units
            too_close = (dists < 2.0) & mask
            loss_close = ((2.0 - dists[too_close]) ** 2).sum()

            # Penalty for max distance per agent exceeding 100 units
            max_dists_per_agent, _ = dists.max(dim=1)
            excess_distance = (max_dists_per_agent - 100.0).clamp(min=0)
            loss_far = (excess_distance ** 2).sum()

            # Bonus: keep agents near origin or somethin
            center = torch.zeros_like(x)
            loss_center = ((next_positions - center) ** 2).mean()

            loss = loss_close + loss_far + 0.1 * loss_center
            loss.backward()
            optimizer.step()

            current_loss = loss.item()
            dx = dx.detach().numpy()

        else:
            with torch.no_grad():
                dx = gnn_model(x, edge_index).numpy()
                current_loss = None

        self.velocities += dx * 0.1
        self.velocities = np.clip(self.velocities, -0.5, 0.5)
        self.positions += self.velocities

env = DummyEnv()
paused = False
current_loss = None
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def interact(event) -> None:
    global paused
    match event.key:
        case ' ':
            paused = not paused
            print('Paused' if paused else 'Running')
        case 'r':
            print("Reset")
            env.reset()
        case 'k':
            print("kill")
            env.kill_first_node()
        case _:
            pass


def update_axes_limits_dynamic(ax, arrays, margin=1.0):
    all_data = np.concatenate(arrays, axis=0)
    mins = np.min(all_data, axis=0) - margin
    maxs = np.max(all_data, axis=0) + margin

    ax.set_xscale('linear')
    ax.set_yscale('linear')

    # in matplotlib do not have set_zscale,

    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])


def main():
    if args.show:
        scat_agents = ax.scatter(env.positions[:, 0], env.positions[:, 1], env.positions[:, 2], c='blue', s=100)
        update_axes_limits_dynamic(ax, [env.positions, env.food], margin=1.0)

        loss_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)
        frame_counter = {'count': 0}

        state = {
            'scat_agents': scat_agents,
            'prev_num_agents': env.positions.shape[0]
        }

        def animate(frame):
            global paused, current_loss
            if not paused:
                env.step(train=True)

                if env.positions.shape[0] != state['prev_num_agents']:
                    state['scat_agents'].remove()
                    state['scat_agents'] = ax.scatter(env.positions[:, 0], env.positions[:, 1], env.positions[:, 2], c='blue', s=100)
                    state['prev_num_agents'] = env.positions.shape[0]
                else:
                    state['scat_agents']._offsets3d = (env.positions[:, 0], env.positions[:, 1], env.positions[:, 2])

                update_axes_limits_dynamic(ax, [env.positions, env.food], margin=1.0)

                if current_loss is not None:
                    loss_text.set_text(f"Loss: {current_loss:.4f}")
                    if frame_counter['count'] % 100 == 0:
                        print(f"Frame {frame}: Loss = {current_loss:.4f}")

                frame_counter['count'] += 1

        fig.canvas.mpl_connect('key_press_event', interact)
        ani = FuncAnimation(fig, animate, interval=args.interval, cache_frame_data=False)
        plt.show()

    else:
        for i in range(10000):
            env.step(train=True)
            if i % 1000 == 0:
                print(f"Training step {i}, Loss: {current_loss}")



if __name__ == '__main__':
    main()
