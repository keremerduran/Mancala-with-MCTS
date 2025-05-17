import json
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from NN import NeuralNet
import time
import multiprocessing


class EarlyStopping:
    def __init__(self, patience=2, delta=0.0, restore_best=True):
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.counter = 0
        self.best_state = None
        self.restore_best = restore_best

    def step(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best:
                self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


class GamesDataset(Dataset):
    def __init__(self, game_states, action_probabilities, win_probabilities):
        self.game_states = game_states
        self.action_probabilities = action_probabilities
        self.win_probabilities = win_probabilities

    def __len__(self):
        return len(self.game_states)

    def __getitem__(self, index):
        return {
            'input': self.game_states[index],
            'action_probabilities': self.action_probabilities[index].view(-1),
            'win_probabilities': self.win_probabilities[index].view(-1)
        }



def loss_function(target_p, pred_p, target_v, pred_v):
    val_loss = F.mse_loss(pred_v, target_v)
    policy_loss = F.cross_entropy(pred_p, target_p)
    return val_loss + policy_loss


def main():
    start_time = time.time()

    file_path  = r'C:\Users\Bogazici\Desktop\MCTS with NNs\games\10_05_25_8000g_40s_MANGO_fourthtraining_trialgames.json'
    model_path = ' '

    with open(file_path2, 'r') as f:
        data = json.load(f)

    game_states, action_probs, win_probs = zip(*data)
    N = 24
    game_states = torch.tensor([[min(N, s)/N for s in st] for st in game_states], dtype=torch.float32)
    action_probs = torch.tensor(action_probs, dtype=torch.float32)
    win_probs    = torch.tensor(win_probs, dtype=torch.float32)

    # Dataset & split
    full_ds = GamesDataset(game_states, action_probs, win_probs)
    n_total = len(full_ds)
    n_val   = int(0.10 * n_total)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # Device & model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralNet()  # ensure dropout layers are added in NN.py if desired
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)

    # Optimizer with stronger weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-3)
    # Scheduler to reduce LR on plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True)

    # Early stopping
    early_stopper = EarlyStopping(patience=2, delta=0.0)

    num_epochs = 8
    for epoch in range(1, num_epochs+1):
        # Training
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x = batch['input'].to(device)
            tp = batch['action_probabilities'].to(device)
            tv = batch['win_probabilities'].to(device)
            pp, vv = model(x)
            loss = loss_function(tp, pp, tv, vv)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train = train_loss / len(train_loader)


        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch['input'].to(device)
                tp = batch['action_probabilities'].to(device)
                tv = batch['win_probabilities'].to(device)
                pp, vv = model(x)
                val_loss += loss_function(tp, pp, tv, vv).item()
        avg_val = val_loss / len(val_loader)

        print(f"Epoch {epoch}/{num_epochs}  "
              f"Train Loss: {avg_train:.4f}  Val Loss: {avg_val:.4f}")


        scheduler.step(avg_val)
        if early_stopper.step(avg_val, model):
            print(f"Early stopping triggered at epoch {epoch}")
            early_stopper.restore(model)
            break

    save_path = '  '
    torch.save(model.state_dict(), save_path)
    print(f"--- {time.time() - start_time:.1f} sec ---")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
