import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributions as dist

class px(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim, args):
        super(px, self).__init__()

        self.n_feature = args.n_feature

        self.fc1 = nn.Sequential(nn.Linear(zd_dim + zx_dim + zy_dim, 128, bias=False), nn.BatchNorm1d(128), nn.ReLU())

        self.un1 = nn.MaxUnpool2d(kernel_size=(1, 2), stride=2)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=(1, 1)),
            nn.ReLU()
        )

        self.un2 = nn.MaxUnpool2d(kernel_size=(1, 2), stride=2)
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=512, kernel_size=(1, 1)),
            nn.ReLU()
        )

        self.un3 = nn.MaxUnpool2d(kernel_size=(1, 2), stride=2)
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=1024, kernel_size=(1, 1)),
            nn.ReLU()
        )

        self.un4 = nn.MaxUnpool2d(kernel_size=(1, 2), stride=2)
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024, out_channels=self.n_feature, kernel_size=(1, 5)),
            nn.ReLU()
        )

        torch.nn.init.xavier_uniform_(self.fc1[0].weight)
        torch.nn.init.xavier_uniform_(self.deconv1[0].weight)
        torch.nn.init.xavier_uniform_(self.deconv2[0].weight)
        torch.nn.init.xavier_uniform_(self.deconv3[0].weight)
        torch.nn.init.xavier_uniform_(self.deconv4[0].weight)

    def forward(self, zd, zx, zy, idxs, sizes):
        if zx is None:
            zdzxzy = torch.cat((zd, zy), dim=-1)
        else:
            zdzxzy = torch.cat((zd, zx, zy), dim=-1)
        h = self.fc1(zdzxzy)
        h = h.view(-1, 64, 1, 2)

        out_1 = self.un1(h, idxs[3], output_size=sizes[2])
        out_11 = self.deconv1(out_1)

        out_2 = self.un2(out_11, idxs[2], output_size=sizes[1])
        out_22 = self.deconv2(out_2)

        out_3 = self.un3(out_22, idxs[1], output_size=sizes[0])
        out_33 = self.deconv3(out_3)

        out_4 = self.un4(out_33, idxs[0])
        out_44 = self.deconv4(out_4)
        out = out_44.permute(0, 2, 3, 1)
        return out


class pzd(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim, args):
        super(pzd, self).__init__()
        self.d_dim = d_dim
        self.device = args.device
        self.now_target_domain_int = int(args.target_domain[-1]) - 1

        self.fc1 = nn.Sequential(nn.Linear(d_dim, zd_dim, bias=False), nn.BatchNorm1d(zd_dim), nn.ReLU())
        self.fc21 = nn.Sequential(nn.Linear(zd_dim, zd_dim))
        self.fc22 = nn.Sequential(nn.Linear(zd_dim, zd_dim), nn.Softplus())

        torch.nn.init.xavier_uniform_(self.fc1[0].weight)
        torch.nn.init.xavier_uniform_(self.fc21[0].weight)
        self.fc21[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc22[0].weight)
        self.fc22[0].bias.data.zero_()

    def forward(self, d):
        d_onehot = torch.zeros(d.shape[0], self.d_dim)
        for idx, val in enumerate(d):
            d_onehot[idx][val.item()] = 1
        d_onehot = d_onehot.to(self.device)
        hidden = self.fc1(d_onehot)
        zd_loc = self.fc21(hidden)
        zd_scale = self.fc22(hidden) + 1e-7
        return zd_loc, zd_scale


class pzy(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim, args):
        super(pzy, self).__init__()

        self.y_dim = y_dim
        self.device = args.device

        self.fc1 = nn.Sequential(nn.Linear(y_dim, zy_dim, bias=False), nn.BatchNorm1d(zy_dim), nn.ReLU())
        self.fc21 = nn.Sequential(nn.Linear(zy_dim, zy_dim))
        self.fc22 = nn.Sequential(nn.Linear(zy_dim, zy_dim), nn.Softplus())

        torch.nn.init.xavier_uniform_(self.fc1[0].weight)
        torch.nn.init.xavier_uniform_(self.fc21[0].weight)
        self.fc21[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc22[0].weight)
        self.fc22[0].bias.data.zero_()

    def forward(self, y):
        y_onehot = torch.zeros(y.shape[0], self.y_dim)
        for idx, val in enumerate(y):
            y_onehot[idx][val.item()] = 1

        y_onehot = y_onehot.to(self.device)

        hidden = self.fc1(y_onehot)
        zy_loc = self.fc21(hidden)
        zy_scale = self.fc22(hidden) + 1e-7

        return zy_loc, zy_scale


# Encoders
class qzd(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim, args):
        super(qzd, self).__init__()

        self.n_feature = args.n_feature

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.n_feature, out_channels=1024, kernel_size=(1, 5)),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=2, return_indices=True, ceil_mode=True)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(1, 1)),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=2, return_indices=True, ceil_mode=True)

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1)),
            nn.ReLU()
        )
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 2), stride=2, return_indices=True, ceil_mode=True)

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1)),
            nn.ReLU()
        )
        self.pool4 = nn.MaxPool2d(kernel_size=(1, 2), stride=2, return_indices=True, ceil_mode=True)

        self.fc11 = nn.Sequential(nn.Linear(128, zd_dim))
        self.fc12 = nn.Sequential(nn.Linear(128, zd_dim), nn.Softplus())

        torch.nn.init.xavier_uniform_(self.conv1[0].weight)
        torch.nn.init.xavier_uniform_(self.conv2[0].weight)
        torch.nn.init.xavier_uniform_(self.conv3[0].weight)
        torch.nn.init.xavier_uniform_(self.conv4[0].weight)

        torch.nn.init.xavier_uniform_(self.fc11[0].weight)
        self.fc11[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc12[0].weight)
        self.fc12[0].bias.data.zero_()

    def forward(self, x):
        x_img = x.float()
        x_img = x_img.view(-1, x_img.shape[3], 1, x_img.shape[2])

        out_conv1 = self.conv1(x_img)
        out1, idx1 = self.pool1(out_conv1)

        out_conv2 = self.conv2(out1)
        out2, idx2 = self.pool2(out_conv2)

        out_conv3 = self.conv3(out2)
        out3, idx3 = self.pool3(out_conv3)

        out_conv4 = self.conv4(out3)
        out4, idx4 = self.pool4(out_conv4)

        out = out4.reshape(-1, out4.shape[1] * out4.shape[3])
        size1 = out1.size()
        size2 = out2.size()
        size3 = out3.size()
        size4 = out4.size()

        zd_loc = self.fc11(out)
        zd_scale = self.fc12(out) + 1e-7

        return zd_loc, zd_scale, [idx1, idx2, idx3, idx4], [size1, size2, size3, size4]


class qzx(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim, args):
        super(qzx, self).__init__()
        self.n_feature = args.n_feature

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.n_feature, out_channels=1024, kernel_size=(1, 5)),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=2, return_indices=True, ceil_mode=True)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(1, 1)),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=2, return_indices=True, ceil_mode=True)

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1)),
            nn.ReLU()
        )
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 2), stride=2, return_indices=True, ceil_mode=True)

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1)),
            nn.ReLU()
        )
        self.pool4 = nn.MaxPool2d(kernel_size=(1, 2), stride=2, return_indices=True, ceil_mode=True)

        self.fc11 = nn.Sequential(nn.Linear(128, zx_dim))
        self.fc12 = nn.Sequential(nn.Linear(128, zx_dim), nn.Softplus())

        torch.nn.init.xavier_uniform_(self.conv1[0].weight)
        torch.nn.init.xavier_uniform_(self.conv2[0].weight)
        torch.nn.init.xavier_uniform_(self.conv3[0].weight)
        torch.nn.init.xavier_uniform_(self.conv4[0].weight)

        torch.nn.init.xavier_uniform_(self.fc11[0].weight)
        self.fc11[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc12[0].weight)
        self.fc12[0].bias.data.zero_()

    def forward(self, x):

        x_img = x.float()
        x_img = x_img.view(-1, x_img.shape[3], 1, x_img.shape[2])

        out_conv1 = self.conv1(x_img)
        out1, idx1 = self.pool1(out_conv1)

        out_conv2 = self.conv2(out1)
        out2, idx2 = self.pool2(out_conv2)

        out_conv3 = self.conv3(out2)
        out3, idx3 = self.pool3(out_conv3)

        out_conv4 = self.conv4(out3)
        out4, idx4 = self.pool4(out_conv4)

        out = out4.reshape(-1, out4.shape[1] * out4.shape[3])
        size1 = out1.size()
        size2 = out2.size()
        size3 = out3.size()
        size4 = out4.size()

        zx_loc = self.fc11(out)
        zx_scale = self.fc12(out) + 1e-7

        return zx_loc, zx_scale, [idx1, idx2, idx3, idx4], [size1, size2, size3, size4]


class qzy(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim, args):
        super(qzy, self).__init__()
        self.n_feature = args.n_feature

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.n_feature, out_channels=1024, kernel_size=(1, 5)),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=2, return_indices=True, ceil_mode=True)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(1, 1)),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=2, return_indices=True, ceil_mode=True)

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1)),
            nn.ReLU()
        )
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 2), stride=2, return_indices=True, ceil_mode=True)

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1)),
            nn.ReLU()
        )
        self.pool4 = nn.MaxPool2d(kernel_size=(1, 2), stride=2, return_indices=True, ceil_mode=True)

        self.fc11 = nn.Sequential(nn.Linear(128, zy_dim))
        self.fc12 = nn.Sequential(nn.Linear(128, zy_dim), nn.Softplus())

        torch.nn.init.xavier_uniform_(self.conv1[0].weight)
        torch.nn.init.xavier_uniform_(self.conv2[0].weight)
        torch.nn.init.xavier_uniform_(self.conv3[0].weight)
        torch.nn.init.xavier_uniform_(self.conv4[0].weight)

        torch.nn.init.xavier_uniform_(self.fc11[0].weight)
        self.fc11[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc12[0].weight)
        self.fc12[0].bias.data.zero_()

    def forward(self, x):

        x_img = x.float()
        x_img = x_img.view(-1, x_img.shape[3], 1, x_img.shape[2])

        out_conv1 = self.conv1(x_img)
        out1, idx1 = self.pool1(out_conv1)

        out_conv2 = self.conv2(out1)
        out2, idx2 = self.pool2(out_conv2)

        out_conv3 = self.conv3(out2)
        out3, idx3 = self.pool3(out_conv3)

        out_conv4 = self.conv4(out3)
        out4, idx4 = self.pool4(out_conv4)

        out = out4.reshape(-1, out4.shape[1] * out4.shape[3]) # [64, 512]
        size1 = out1.size()
        size2 = out2.size()
        size3 = out3.size()
        size4 = out4.size()

        zy_loc = self.fc11(out)
        zy_scale = self.fc12(out) + 1e-7

        return zy_loc, zy_scale, [idx1, idx2, idx3, idx4], [size1, size2, size3, size4]


# Auxiliary tasks
class qd(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super(qd, self).__init__()

        self.fc1 = nn.Linear(zd_dim, d_dim)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.zero_()

    def forward(self, zd):
        h = F.relu(zd)
        loc_d = self.fc1(h)
        return loc_d


class qy(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super(qy, self).__init__()

        self.fc1 = nn.Linear(zy_dim, y_dim)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.zero_()

    def forward(self, zy):
        h = F.relu(zy)
        loc_y = self.fc1(h)

        return loc_y


class GILE(nn.Module):
    def __init__(self, args):
        super(GILE, self).__init__()
        self.zd_dim = args.d_AE
        self.zx_dim = 0
        self.zy_dim = args.d_AE
        self.d_dim = args.n_domains
        self.x_dim = args.x_dim
        self.y_dim = args.n_class

        self.start_zx = self.zd_dim
        self.start_zy = self.zd_dim + self.zx_dim

        self.px = px(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim, args)
        self.pzd = pzd(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim, args)
        self.pzy = pzy(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim, args)

        self.qzd = qzd(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim, args)
        if self.zx_dim != 0:
            self.qzx = qzx(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim, args)
        self.qzy = qzy(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim, args)

        self.qd = qd(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim)
        self.qy = qy(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim)

        self.aux_loss_multiplier_y = args.aux_loss_multiplier_y
        self.aux_loss_multiplier_d = args.aux_loss_multiplier_d

        self.beta_d = args.beta_d
        self.beta_x = args.beta_x
        self.beta_y = args.beta_y

        self.cuda()

    def forward(self, d, x, y):
        x = torch.unsqueeze(x, 1)

        d = d.long()
        y = y.long()

        # Encode
        zd_q_loc, zd_q_scale, _, _ = self.qzd(x)
        if self.zx_dim != 0:
            zx_q_loc, zx_q_scale, _, _ = self.qzx(x)
        zy_q_loc, zy_q_scale, idxs_y, sizes_y = self.qzy(x)

        # Reparameterization trick
        qzd = dist.Normal(zd_q_loc, zd_q_scale)
        zd_q = qzd.rsample()
        if self.zx_dim != 0:
            qzx = dist.Normal(zx_q_loc, zx_q_scale)
            zx_q = qzx.rsample()
        else:
            qzx = None
            zx_q = None

        qzy = dist.Normal(zy_q_loc, zy_q_scale)
        zy_q = qzy.rsample()

        # Decode
        x_recon = self.px(zd_q, zx_q, zy_q, idxs_y, sizes_y)

        zd_p_loc, zd_p_scale = self.pzd(d)

        if self.zx_dim != 0:
            zx_p_loc, zx_p_scale = torch.zeros(zd_p_loc.size()[0], self.zx_dim).cuda(),\
                                   torch.ones(zd_p_loc.size()[0], self.zx_dim).cuda()
        zy_p_loc, zy_p_scale = self.pzy(y)

        # Reparameterization trick
        pzd = dist.Normal(zd_p_loc, zd_p_scale)
        if self.zx_dim != 0:
            pzx = dist.Normal(zx_p_loc, zx_p_scale)
        else:
            pzx = None
        pzy = dist.Normal(zy_p_loc, zy_p_scale)

        # Auxiliary losses
        d_hat = self.qd(zd_q)
        y_hat = self.qy(zy_q)

        return x_recon, d_hat, y_hat, qzd, pzd, zd_q, qzx, pzx, zx_q, qzy, pzy, zy_q

    def loss_function_false(self, args, d, x, y=None):
        d = d.long()
        y = y.long()

        pred_d, pred_y, pred_d_false, pred_y_false = self.classifier(x)

        loss_classify_true = args.weight_true * (F.cross_entropy(pred_d, d, reduction='sum') + F.cross_entropy(pred_y, y, reduction='sum'))
        loss_classify_false = args.weight_false * (F.cross_entropy(pred_d_false, d, reduction='sum') + F.cross_entropy(pred_y_false, y, reduction='sum'))

        loss = loss_classify_true - loss_classify_false

        loss.requires_grad = True

        return loss

    def loss_function(self, d, x, y=None):
        d = d.long()
        y = y.long()

        x_recon, d_hat, y_hat, qzd, pzd, zd_q, qzx, pzx, zx_q, qzy, pzy, zy_q = self.forward(d, x, y)

        x = torch.unsqueeze(x, 1)
        CE_x = F.mse_loss(x_recon, x.float())

        zd_p_minus_zd_q = torch.sum(pzd.log_prob(zd_q) - qzd.log_prob(zd_q))
        if self.zx_dim != 0:
            KL_zx = torch.sum(pzx.log_prob(zx_q) - qzx.log_prob(zx_q))
        else:
            KL_zx = 0

        zy_p_minus_zy_q = torch.sum(pzy.log_prob(zy_q) - qzy.log_prob(zy_q))
        CE_d = F.cross_entropy(d_hat, d, reduction='sum')
        CE_y = F.cross_entropy(y_hat, y, reduction='sum')

        return CE_x \
               - self.beta_d * zd_p_minus_zd_q \
               - self.beta_x * KL_zx \
               - self.beta_y * zy_p_minus_zy_q \
               + self.aux_loss_multiplier_d * CE_d \
               + self.aux_loss_multiplier_y * CE_y,\
               CE_y

    def classifier(self, x):
        """
        classify an image (or a batch of images)
        :param xs: a batch of scaled vectors of pixels from an image
        :return: a batch of the corresponding class labels (as one-hots)
        """
        with torch.no_grad():

            x = torch.unsqueeze(x, 1)

            zd_q_loc, zd_q_scale, _, _ = self.qzd(x)
            zd = zd_q_loc
            alpha = F.softmax(self.qd(zd), dim=1)

            # get the index (digit) that corresponds to
            # the maximum predicted class probability
            res, ind = torch.topk(alpha, 1)

            # convert the digit(s) to one-hot tensor(s)
            d = x.new_zeros(alpha.size())
            d = d.scatter_(1, ind, 1.0)

            zy_q_loc, zy_q_scale, _, _ = self.qzy.forward(x)
            zy = zy_q_loc
            alpha = F.softmax(self.qy(zy), dim=1)

            # get the index (digit) that corresponds to
            # the maximum predicted class probability
            res, ind = torch.topk(alpha, 1)

            # convert the digit(s) to one-hot tensor(s)
            y = x.new_zeros(alpha.size())
            y = y.scatter_(1, ind, 1.0)

            alpha_y2d = F.softmax(self.qd(zy), dim=1)

            # get the index (digit) that corresponds to
            # the maximum predicted class probability
            res, ind = torch.topk(alpha_y2d, 1)
            # convert the digit(s) to one-hot tensor(s)
            d_false = x.new_zeros(alpha_y2d.size())
            d_false = d_false.scatter_(1, ind, 1.0)

            alpha_d2y = F.softmax(self.qy(zd), dim=1)

            # get the index (digit) that corresponds to
            # the maximum predicted class probability
            res, ind = torch.topk(alpha_d2y, 1)

            # convert the digit(s) to one-hot tensor(s)
            y_false = x.new_zeros(alpha_d2y.size())
            y_false = y_false.scatter_(1, ind, 1.0)

        return d, y, d_false, y_false

    def get_features(self, x):
        zy_q_loc, zy_q_scale, idxs_y, sizes_y = self.qzy(x)
        qzy = dist.Normal(zy_q_loc, zy_q_scale)
        zy_q = qzy.rsample()
        return zy_q



