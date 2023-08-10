'''
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License
https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

Copyright (c) 2023, Takanori Fujiwara and S. Sandra Bae
All rights reserved.
'''

import numpy as np
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class LayoutAdjustment():

    def __init__(
            self,
            link_radius,
            node_radius,
            ref_node_positions=None,
            non_intersect_prior_links=None,
            n_components=3,
            encoder=None,
            learning_rate=1e-3,
            batch_size=None,
            loss_weights={
                'layout_change': 0,
                'prior_link_intersect': 0,
                'link_intersect': 0,
                'node_intersect': 0,
                'link_length_variation': 0
            },
            device='cpu'):
        self.link_radius = link_radius
        self.node_radius = node_radius
        self.ref_node_positions = ref_node_positions
        if isinstance(self.ref_node_positions, np.ndarray):
            self.ref_node_positions = torch.from_numpy(
                ref_node_positions).float()
        self.non_intersect_prior_links = non_intersect_prior_links
        self.n_components = n_components
        self.encoder = encoder
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.loss_weights = loss_weights
        self.device = device

    def fit(self, nodes, links, max_epochs=100):
        if self.batch_size is None:
            self.batch_size = min(len(nodes), 1000)

        I = torch.eye(len(nodes))
        dataset = CustomDataset(torch.Tensor(links).long())

        trainer = pl.Trainer(accelerator=self.device,
                             devices='auto',
                             max_epochs=max_epochs)

        if self.encoder is None:
            self.encoder = DefaultEncoder(len(nodes),
                                          n_components=self.n_components)

        self.model = Model(
            learning_rate=self.learning_rate,
            encoder=self.encoder,
            device=self.device,
            link_radius=self.link_radius,
            node_radius=self.node_radius,
            I=I,
            ref_node_positions=self.ref_node_positions,
            non_intersect_prior_links=self.non_intersect_prior_links,
            loss_weights=self.loss_weights)

        trainer.fit(model=self.model,
                    datamodule=DataModule(dataset, self.batch_size))

    @torch.no_grad()
    def transform(self, nodes, axes_adjustment=True):
        I = torch.eye(len(nodes))
        node_positions = self.model.encoder(I).detach().cpu().numpy()

        if axes_adjustment:
            # PCA-based axes adjustment
            # reasoning: along 3rd PC direction/height-direction,
            #            variance/position differences would be smallest
            pca = PCA(n_components=3)
            node_positions = pca.fit_transform(node_positions -
                                               node_positions.mean(axis=0))

        return node_positions


class DataModule(pl.LightningDataModule):

    def __init__(self, dataset, batch_size):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size

    def train_dataloader(self) -> DataLoader:
        # num_workers should be 0 as we are accessing to one array/tensor
        return DataLoader(dataset=self.dataset,
                          batch_size=self.batch_size,
                          num_workers=0,
                          drop_last=True)


class CustomDataset(Dataset):

    def __init__(self, links):
        shuffled_indices = np.random.permutation(len(links))
        self.links = links[shuffled_indices]

    def __len__(self):
        return len(self.links)

    def __getitem__(self, index):
        return self.links[index]


class Model(pl.LightningModule):

    def __init__(self,
                 learning_rate,
                 encoder,
                 link_radius,
                 node_radius,
                 I,
                 ref_node_positions,
                 non_intersect_prior_links,
                 loss_weights,
                 device='cpu'):
        super().__init__()

        if device == 'auto':
            self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self._device = device

        self.lr = learning_rate
        self.encoder = encoder.to(self._device)
        self.link_radius = torch.Tensor([link_radius])
        self.node_radius = torch.Tensor([node_radius])
        self.I = I
        self.ref_node_positions = ref_node_positions
        self.non_intersect_prior_links = non_intersect_prior_links
        self.loss_weights = loss_weights

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        links = batch
        node_positions = self.encoder(self.I)

        loss = Loss(self._device)

        layout_change_loss = 0
        if self.loss_weights.get('layout_change') and (self.ref_node_positions
                                                       is not None):
            layout_change_loss = loss.layout_change_loss(
                node_positions, ref_node_positions=self.ref_node_positions
            ) * self.loss_weights['layout_change']

        prior_link_intersect_loss = 0
        if self.loss_weights.get('prior_link_intersect') and (
                self.non_intersect_prior_links is not None):
            prior_link_intersect_loss = loss.link_intersection_loss(
                self.non_intersect_prior_links,
                link_radius=self.link_radius,
                node_radius=self.node_radius,
                node_positions=node_positions
            ) * self.loss_weights['prior_link_intersect']

        link_intersect_loss = 0
        if self.loss_weights.get('link_intersect'):
            link_intersect_loss = loss.link_intersection_loss(
                links,
                link_radius=self.link_radius,
                node_radius=self.node_radius,
                node_positions=node_positions
            ) * self.loss_weights['link_intersect']

        node_intersect_loss = 0
        if self.loss_weights.get('node_intersect'):
            node_intersect_loss = loss.node_intesection_loss(
                node_positions, node_radius=self.node_radius
            ) * self.loss_weights['node_intersect']

        link_length_variation_loss = 0
        if self.loss_weights.get('link_length_variation'):
            link_length_variation_loss = loss.link_length_variation_loss(
                links, node_positions=node_positions
            ) * self.loss_weights['link_length_variation']

        # non_horizontal_loss = loss.non_horizontal_loss(node_positions)

        encoder_loss = layout_change_loss + prior_link_intersect_loss + link_intersect_loss + node_intersect_loss + link_length_variation_loss

        self.log('loss', encoder_loss, prog_bar=True)

        return encoder_loss


class DefaultEncoder(nn.Module):

    def __init__(self, dims, n_components=3, hidden_size=100):
        super(DefaultEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.product(dims), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_components),
        )

    def forward(self, x):
        return self.encoder(x)


class Loss():

    def __init__(self, device='cpu'):
        self.device = device

    def layout_change_loss(self,
                           node_positions,
                           ref_node_positions,
                           loss_fn=F.huber_loss):
        # translation/centering
        P_ref = ref_node_positions - ref_node_positions.mean(axis=0)
        P = node_positions - node_positions.mean(axis=0)

        # uniform scaling
        eps = torch.finfo(torch.float32).eps
        P_ref = P_ref / (torch.norm(P_ref, dim=0).mean() + eps)
        P = P / (torch.norm(P, dim=0).mean() + eps)

        # rotate and reflection
        U, S, Vh = torch.linalg.svd(P_ref.t() @ P)
        R = Vh.t() @ U.t()
        P = P @ R

        # TODO: maybe add normalization by ||x_ref_||
        # loss = F.mse_loss(x_ref_, x_)  # not robust to outliers
        loss = loss_fn(P_ref, P)

        return loss

    def links_intersected(self,
                          link1,
                          link2,
                          link_radius,
                          node_radius,
                          node_positions,
                          eps=1e-16):
        # See http://paulbourke.net/geometry/pointlineplane/ to know details
        p0_ = node_positions[link1[0]]
        p1_ = node_positions[link1[1]]
        p2_ = node_positions[link2[0]]
        p3_ = node_positions[link2[1]]

        # remove the link related the node's radius part
        # Note: this is not perfect as the node shape is sphere
        u1 = (p1_ - p0_) / ((p1_ - p0_).norm() + eps)
        u2 = (p3_ - p2_) / ((p3_ - p2_).norm() + eps)
        p0 = p0_ + u1 * node_radius
        p1 = p1_ - u1 * node_radius
        p2 = p2_ + u2 * node_radius
        p3 = p3_ - u2 * node_radius

        d0232 = ((p0 - p2) * (p3 - p2)).sum()
        d3210 = ((p3 - p2) * (p1 - p0)).sum()
        d0210 = ((p0 - p2) * (p1 - p0)).sum()
        d3232 = ((p3 - p2) * (p3 - p2)).sum()
        d1010 = ((p1 - p0) * (p1 - p0)).sum()

        mua = (d0232 * d3210 - d0210 * d3232) / (d1010 * d3232 -
                                                 d3210 * d3210 + eps)
        mub = (d0232 + mua * d3210) / (d3232 + eps)

        # limit in the line length
        # should be 0 <= mua, mub <=1
        mua = torch.min(torch.Tensor([1.0]), torch.max(torch.Tensor([0.0]),
                                                       mua))
        mub = torch.min(torch.Tensor([1.0]), torch.max(torch.Tensor([0.0]),
                                                       mub))
        pa = p0 + mua * (p1 - p0)
        pb = p2 + mub * (p3 - p2)
        dist = (pa - pb).norm()

        return F.relu(link_radius * 2 - dist)  # if dist > link_radius * 2: 0

    def link_intersection_loss(self,
                               links,
                               link_radius,
                               node_radius,
                               node_positions,
                               eps=1e-16):
        loss = 0
        for i in range(len(links)):
            link1 = links[i]
            for j in range(i + 1, len(links)):
                link2 = links[j]
                loss += self.links_intersected(link1,
                                               link2,
                                               link_radius=link_radius,
                                               node_radius=node_radius,
                                               node_positions=node_positions,
                                               eps=eps)

        return loss

    def nodes_intersected(self, s_pos, t_pos, node_radius):
        dist = (s_pos - t_pos).norm()
        return F.relu(node_radius * 2 - dist)  # if dist > node_radius * 2: 0

    def node_intesection_loss(self, node_positions, node_radius):
        loss = 0
        for i in range(len(node_positions)):
            s_pos = node_positions[i]
            for j in range(i + 1, len(node_positions)):
                t_pos = node_positions[j]
                loss += self.nodes_intersected(s_pos,
                                               t_pos,
                                               node_radius=node_radius)

        return loss

    def link_length_variation_loss(self, links, node_positions):
        P = node_positions[links, :]  # shape (n_links, s and t, 3d)
        link_lengths = (P[:, 0, :] - P[:, 1, :]).norm(dim=1)
        loss = link_lengths.std()

        return loss

    def non_horizontal_loss(self, node_positions):
        loss = (node_positions[:, 2] / node_positions.norm(dim=1)).norm()

        return loss


if __name__ == '__main__':
    # four node graphlet example
    nodes = [0, 1, 2, 3]
    links = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]

    link_radius = 3.15  # 3.15mm
    node_radius = 8.5  # 8.5mm
    mean_link_length = 50

    la = LayoutAdjustment(link_radius=link_radius,
                          node_radius=node_radius,
                          n_components=3,
                          batch_size=len(links),
                          loss_weights={
                              'link_intersect': 1,
                              'node_intersect': 1,
                              'link_length_variation': 1
                          })
    la.fit(nodes, links, max_epochs=500)

    node_positions = la.transform(nodes)
    node_positions *= mean_link_length / np.array([
        np.linalg.norm(node_positions[s] - node_positions[t]) for s, t in links
    ]).mean()

    print(node_positions)