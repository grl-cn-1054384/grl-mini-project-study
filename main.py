import random
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as geom

matplotlib.use("TkAgg")

# Read all recipes from file
with open("recipes.txt") as f:
    recipes = [re.fullmatch(r"(.+) \+ (.+) = (.+)", r.strip()) for r in f.readlines()]

# Compute a list of items used in/produced from the recipes
# This allows us to map each item to a unique number
items = []
for recipe in recipes:
    # recipe[1] and recipe[2] are ingredients, recipe[3] is the result
    for i in range(1, 4):
        if recipe[i] not in items:
            items += [recipe[i]]

# Adjacency matrices
item_to_recipe = torch.zeros([len(recipes), len(items)]).to(torch.int64)
recipe_to_item = torch.zeros([len(items), len(recipes)]).to(torch.int64)

# Populate matrices
for i, recipe in enumerate(recipes):
    item_to_recipe[i, items.index(recipe[1])] = 1
    item_to_recipe[i, items.index(recipe[2])] = 1
    recipe_to_item[items.index(recipe[3]), i] = 1


# Converts an item name into a one-hot encoded vector
def name_to_vec(name, length=None):
    n = items.index(name)
    true_length = len(items) if length is None else length
    return F.one_hot(torch.Tensor([n]).to(torch.int64), true_length)[0].to(torch.float32)


# Compute item costs

item_costs = {
    "air": 0,
    "earth": 0,
    "fire": 0,
    "water": 0
}
recipe_costs = {}

has_updated = True
while has_updated:
    has_updated = False

    for recipe in recipes:
        cost1 = None if recipe[1] not in item_costs else item_costs[recipe[1]]
        cost2 = None if recipe[2] not in item_costs else item_costs[recipe[2]]

        if cost1 is not None and cost2 is not None:
            has_updated = f"{recipe[1]} + {recipe[2]}" in recipe_costs and \
                recipe_costs[f"{recipe[1]} + {recipe[2]}"] != (recipe[3], cost1 + cost2 + 1)
            recipe_costs[f"{recipe[1]} + {recipe[2]}"] = (recipe[3], cost1 + cost2 + 1)

    for item in items:
        costs = [c for (i, c) in recipe_costs.values() if i == item]
        if len(costs) == 0:
            continue

        cost = min(costs)
        if item not in item_costs or cost < item_costs[item]:
            has_updated = True
            item_costs[item] = cost


# Compute vector of all costs for each item
cost_vector = torch.zeros([len(items)])
for (i, c) in item_costs.items():
    cost_vector += c * name_to_vec(i)


# A regular GNN with sum aggregation
class GNNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, act_fn):
        super().__init__()

        self.agg = geom.nn.aggr.SumAggregation()
        self.w_self = nn.Linear(input_dim, output_dim, bias=False)
        self.w_neigh = nn.Linear(input_dim, output_dim, bias=False)
        self.act_fn = act_fn()

    def forward(self, node_feats, adj_matrix):
        neigh_agg = torch.zeros(node_feats.shape)

        for i in range(node_feats.shape[0]):
            agg = self.agg(node_feats.T, adj_matrix[i], dim=1)
            if agg.shape[1] == 2:
                neigh_agg[i] = agg[0][1]

        return self.act_fn(self.w_self(node_feats) + self.w_neigh(neigh_agg))


class GNNModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()

        layers = [GNNLayer(input_dim, hidden_dim, nn.ReLU)]
        for i in range(1, num_layers - 1):
            layers += [GNNLayer(hidden_dim, hidden_dim, nn.ReLU)]
        layers += [GNNLayer(hidden_dim, output_dim, nn.ReLU)]

        self.layers = nn.ModuleList(layers)

    def forward(self, x, adj_matrix):
        for layer in self.layers:
            x = layer(x, adj_matrix)
        return x


# A "dual" GNN with sum aggregation and max aggregation
class GNNDualLayer(nn.Module):
    def __init__(self, input_dim, output_dim, act_fn):
        super().__init__()

        self.agg1 = geom.nn.aggr.MaxAggregation()
        self.w1_self = nn.Linear(input_dim, output_dim, bias=False)
        self.w1_neigh = nn.Linear(input_dim, output_dim, bias=False)
        self.act_fn1 = act_fn()

        self.agg2 = geom.nn.aggr.SumAggregation()
        self.w2_self = nn.Linear(input_dim, output_dim, bias=False)
        self.w2_neigh = nn.Linear(input_dim, output_dim, bias=False)
        self.act_fn2 = act_fn()

    def forward(self, node_feats1, node_feats2, adj_1to2, adj_2to1):
        neigh_agg1 = torch.zeros(node_feats1.shape)
        neigh_agg2 = torch.zeros(node_feats2.shape)

        for i in range(node_feats1.shape[0]):
            agg = self.agg1(node_feats2.T, adj_2to1[i], dim=1)
            if agg.shape[1] == 2:
                neigh_agg1[i] = agg[0][1]

        for i in range(node_feats2.shape[0]):
            agg = self.agg2(node_feats1.T, adj_1to2[i], dim=1)
            if agg.shape[1] == 2:
                neigh_agg2[i] = agg[0][1]

        return (
            self.act_fn1(self.w1_self(node_feats1) + self.w1_neigh(neigh_agg1)),
            self.act_fn2(self.w2_self(node_feats2) + self.w2_neigh(neigh_agg2))
        )


class GNNDualModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()

        layers = [GNNDualLayer(input_dim, hidden_dim, nn.ReLU)]
        for i in range(1, num_layers - 1):
            layers += [GNNDualLayer(hidden_dim, hidden_dim, nn.ReLU)]
        layers += [GNNDualLayer(hidden_dim, output_dim, nn.Identity)]

        self.layers = nn.ModuleList(layers)

    def forward(self, x1, x2, adj_1to2, adj_2to1):
        for layer in self.layers:
            x1, x2 = layer(x1, x2, adj_1to2, adj_2to1)
        return x1, x2


params = {
    "input_features": 1,
    "hidden_features": 3,
    "num_gnn_layers": 20,
    "learning_rate": 1e-3,
    "num_epochs": 600,
}

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model1 = GNNModule(
    params["input_features"],
    params["hidden_features"],
    1,
    num_layers=params["num_gnn_layers"],
).to(device)
model2 = GNNDualModule(
    params["input_features"],
    params["hidden_features"],
    1,
    num_layers=params["num_gnn_layers"],
).to(device)


def train(data, params):
    train_mask = []
    test_mask = []

    for i in range(len(items)):
        if random.randint(1, 5) > 1:
            train_mask += [i]
        else:
            test_mask += [i]

    train_mask = torch.Tensor(train_mask).to(torch.int64).reshape((len(train_mask), 1))
    test_mask = torch.Tensor(test_mask).to(torch.int64).reshape((len(test_mask), 1))

    train_data = data["y"].gather(dim=0, index=train_mask)
    test_data = data["y"].gather(dim=0, index=test_mask)

    adj_full = torch.cat(
        (
            torch.cat((torch.zeros([len(items), len(items)]), recipe_to_item), dim=1),
            torch.cat((item_to_recipe, torch.zeros([len(recipes), len(recipes)])), dim=1)
        ),
        dim=0
    ).to(torch.int64)

    losses1 = []
    losses2 = []
    losses1_test = []
    losses2_test = []

    optimizer1 = torch.optim.Adam(model1.parameters(), lr=params["learning_rate"])
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=params["learning_rate"])
    loss_fn1 = nn.MSELoss()
    loss_fn2 = nn.MSELoss()

    loss_fn1_test = nn.MSELoss()
    loss_fn2_test = nn.MSELoss()

    for i in range(params["num_epochs"]):
        optimizer1.zero_grad()
        optimizer2.zero_grad()

        y1 = model1(data["x"], adj_full)
        y2, _ = model2(data["x"][:len(items)], data["x"][len(items):], item_to_recipe, recipe_to_item)

        y1_train = y1[:len(items)].gather(dim=0, index=train_mask)
        y1_test = y1[:len(items)].gather(dim=0, index=test_mask)
        y2_train = y2[:len(items)].gather(dim=0, index=train_mask)
        y2_test = y2[:len(items)].gather(dim=0, index=test_mask)

        loss1 = loss_fn1(y1_train, train_data)
        loss2 = loss_fn2(y2_train, train_data)

        loss1.backward()
        loss2.backward()
        optimizer1.step()
        optimizer2.step()

        loss1_test = loss_fn1_test(y1_test, test_data)
        loss2_test = loss_fn2_test(y2_test, test_data)

        if i > 0:
            losses1 += [loss1.item()]
            losses2 += [loss2.item()]
            losses1_test += [loss1_test.item()]
            losses2_test += [loss2_test.item()]

        if i % 20 == 0:
            if len(losses1) > 0 and loss1.item() < 50:
                t = np.arange(0, len(losses1), 1)
                plt.clf()
                plt.plot(t, np.array(losses1), label="Vanilla Model")
                plt.plot(t, np.array(losses2), label="Dual Model")
                plt.legend(loc="upper right")
                plt.savefig(f"plots/train_plot{i}")

                t = np.arange(0, len(losses1_test), 1)
                plt.clf()
                plt.plot(t, np.array(losses1_test), label="Vanilla Model")
                plt.plot(t, np.array(losses2_test), label="Dual Model")
                plt.legend(loc="upper right")
                plt.savefig(f"plots/test_plot{i}")

            print("Loss: ", loss1_test.item(), "\t\tDual Loss: ", loss2_test.item())

    return losses1, losses2


if __name__ == "__main__":
    print(cost_vector)

    # Only feature vectors per node is a number indicating
    # whether the node is in the set {air, earth, fire, water}
    losses = train({
        "x": 2 * torch.reshape(
            name_to_vec("air", length=len(items) + len(recipes)) +
            name_to_vec("earth", length=len(items) + len(recipes)) +
            name_to_vec("fire", length=len(items) + len(recipes)) +
            name_to_vec("water", length=len(items) + len(recipes)),
            (len(items) + len(recipes), 1)
        ) - 1,
        "y": torch.reshape(cost_vector, (len(items), 1))
    }, params)

    t = np.arange(0, params["num_epochs"], 1)

    plt.plot(t, np.array(losses))
    plt.show()

    # print(losses)
