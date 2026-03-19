import torch
data = torch.load(r"get_protain_nodes\\test\\data\\train_graphs_for_gan.pt", weights_only=False)
g = data[0]
print(g)
print("has pos:", hasattr(g, 'pos') and g.pos is not None)
print("pos shape:", g.pos.shape if hasattr(g, 'pos') and g.pos is not None else "N/A")