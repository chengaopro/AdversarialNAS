import torch

def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])

class Architect_gen(object):
    def __init__(self, model, args):
        self.args = args
        if isinstance(model, torch.nn.DataParallel):
            self.model = model.module
        else:
            self.model = model
        self.optimizer = torch.optim.Adam(self.model.arch_parameters(), lr=0.0003, betas=(0.5, 0.999), weight_decay=0.0001)

    def step(self, search_z, gen_net, dis_net, train_z=None, eta=None):
        self.optimizer.zero_grad()
        if self.args.amending_coefficient:
            self._backward_step_amending(search_z, gen_net, dis_net, train_z, eta)
        else:
            self._backward_step(search_z, gen_net, dis_net)
        self.optimizer.step()

    def _backward_step(self, search_z, gen_net, dis_net):
        gen_imgs = gen_net(search_z)
        fake_validity = dis_net(gen_imgs)
        # cal loss
        g_loss = -torch.mean(fake_validity)
        g_loss.backward()

    def _backward_step_amending(self, search_z, gen_net, dis_net, train_z, eta):
        gen_imgs = gen_net(search_z)
        fake_validity = dis_net(gen_imgs)
        # cal loss
        g_loss = -torch.mean(fake_validity)
        g_loss.backward()

        vector = [v.grad.data for v in self.model.parameters()]
        implicit_grads = self._hessian_vector_product_2(
            self._hessian_vector_product_1(vector, train_z, gen_net, dis_net), train_z, gen_net, dis_net)

        for g, ig in zip(self.model.arch_parameters(), implicit_grads):
            g.grad.data.sub_(eta, ig.data)

    # Compute Hessian matrix product (codes from https://openreview.net/forum?id=BJlgt2EYwr)
    def _hessian_vector_product_2(self, vector, train_z, gen_net, dis_net, r=1e-2):
        R = r / _concat(vector).norm()
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)
        # cal loss
        gen_imgs = gen_net(train_z)
        fake_validity = dis_net(gen_imgs)
        g_loss = -torch.mean(fake_validity)
        grads_p = torch.autograd.grad(g_loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.sub_(2 * R, v)
        # cal loss
        gen_imgs = gen_net(train_z)
        fake_validity = dis_net(gen_imgs)
        g_loss = -torch.mean(fake_validity)
        grads_n = torch.autograd.grad(g_loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]

    # Compute Hessian matrix product
    def _hessian_vector_product_1(self, vector, train_z, gen_net, dis_net, r=1e-2):
        R = r / _concat(vector).norm()
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)
        # cal loss
        gen_imgs = gen_net(train_z)
        fake_validity = dis_net(gen_imgs)
        g_loss = -torch.mean(fake_validity)

        grads_p = torch.autograd.grad(g_loss, self.model.parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.sub_(2 * R, v)
        # cal loss
        gen_imgs = gen_net(train_z)
        fake_validity = dis_net(gen_imgs)
        g_loss = -torch.mean(fake_validity)
        grads_n = torch.autograd.grad(g_loss, self.model.parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]

# --------------------------------------------------------------------------------------- #

class Architect_dis(object):
    def __init__(self, model, args):
        self.args = args
        if isinstance(model, torch.nn.DataParallel):
            self.model = model.module
        else:
            self.model = model
        self.optimizer = torch.optim.Adam(self.model.arch_parameters(), lr=0.0003, betas=(0.5, 0.999), weight_decay=0.0001)

    def step(self, dis_net, real_imgs, gen_net, search_z, real_imgs_train=None, train_z=None, eta=None):
        self.optimizer.zero_grad()
        if self.args.amending_coefficient:
            self._backward_step_amending(dis_net, real_imgs, gen_net, search_z, real_imgs_train, train_z, eta)
        else:
            self._backward_step(dis_net, real_imgs, gen_net, search_z)
        self.optimizer.step()

    def _backward_step(self, dis_net, real_imgs, gen_net, search_z):
        real_validity = dis_net(real_imgs)
        fake_imgs = gen_net(search_z).detach()
        fake_validity = dis_net(fake_imgs)
        # cal loss
        d_loss = torch.mean(torch.nn.ReLU(inplace=True)(1.0 - real_validity)) + \
                 torch.mean(torch.nn.ReLU(inplace=True)(1 + fake_validity))
        d_loss.backward()

    def _backward_step_amending(self, dis_net, real_imgs, gen_net, search_z, real_imgs_train, train_z, eta):
        real_validity = dis_net(real_imgs)
        fake_imgs = gen_net(search_z).detach()
        fake_validity = dis_net(fake_imgs)
        # cal loss
        d_loss = torch.mean(torch.nn.ReLU(inplace=True)(1.0 - real_validity)) + \
                 torch.mean(torch.nn.ReLU(inplace=True)(1 + fake_validity))
        d_loss.backward()

        vector = [v.grad.data for v in self.model.parameters()]
        implicit_grads = self._hessian_vector_product_2(
            self._hessian_vector_product_1(vector, real_imgs_train, train_z, gen_net, dis_net), real_imgs_train,
            train_z, gen_net, dis_net)

        for g, ig in zip(self.model.arch_parameters(), implicit_grads):
            g.grad.data.sub_(eta, ig.data)

    # Compute Hessian matrix product (codes from https://openreview.net/forum?id=BJlgt2EYwr)
    def _hessian_vector_product_2(self, vector, real_imgs_train, train_z, gen_net, dis_net, r=1e-2):
        R = r / _concat(vector).norm()
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)
        # cal loss
        real_validity = dis_net(real_imgs_train)
        fake_imgs = gen_net(train_z).detach()
        fake_validity = dis_net(fake_imgs)
        d_loss = torch.mean(torch.nn.ReLU(inplace=True)(1.0 - real_validity)) + \
                 torch.mean(torch.nn.ReLU(inplace=True)(1 + fake_validity))

        grads_p = torch.autograd.grad(d_loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.sub_(2 * R, v)
        # cal loss
        real_validity = dis_net(real_imgs_train)
        fake_imgs = gen_net(train_z).detach()
        fake_validity = dis_net(fake_imgs)
        d_loss = torch.mean(torch.nn.ReLU(inplace=True)(1.0 - real_validity)) + \
                 torch.mean(torch.nn.ReLU(inplace=True)(1 + fake_validity))

        grads_n = torch.autograd.grad(d_loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]

    # Compute Hessian matrix product
    def _hessian_vector_product_1(self, vector, real_imgs_train, train_z, gen_net, dis_net, r=1e-2):
        R = r / _concat(vector).norm()
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)
        # cal loss
        real_validity = dis_net(real_imgs_train)
        fake_imgs = gen_net(train_z).detach()
        fake_validity = dis_net(fake_imgs)
        d_loss = torch.mean(torch.nn.ReLU(inplace=True)(1.0 - real_validity)) + \
                 torch.mean(torch.nn.ReLU(inplace=True)(1 + fake_validity))

        grads_p = torch.autograd.grad(d_loss, self.model.parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.sub_(2 * R, v)
        # cal loss
        real_validity = dis_net(real_imgs_train)
        fake_imgs = gen_net(train_z).detach()
        fake_validity = dis_net(fake_imgs)
        d_loss = torch.mean(torch.nn.ReLU(inplace=True)(1.0 - real_validity)) + \
                 torch.mean(torch.nn.ReLU(inplace=True)(1 + fake_validity))

        grads_n = torch.autograd.grad(d_loss, self.model.parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]
