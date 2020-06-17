# @Date    : 2019-10-22
# @Author  : Chen Gao

import torch.nn.functional as F
import numpy as np


PRIMITIVES = [
  'none',
  'skip_connect',
  'conv_1x1',
  'conv_3x3',
  'conv_5x5',
  'dil_conv_3x3',
  'dil_conv_5x5'
]

PRIMITIVES_wo_act = [
  'conv_1x1',
  'conv_3x3',
  'conv_5x5',
  'dil_conv_3x3',
  'dil_conv_5x5'
]

PRIMITIVES_up = [
  'nearest',
  'bilinear',
  'ConvTranspose'
]

PRIMITIVES_down = [
  'avg_pool',
  'max_pool',
  'conv_3x3',
  'conv_5x5',
  'dil_conv_3x3',
  'dil_conv_5x5'
]


def alpha2genotype(alpha_normal, alpha_up, save=False, file_path=None):
    num_cell = alpha_up.shape[0]
    offset = alpha_up.shape[1]

    genotype = np.ones((alpha_up.shape[0], alpha_up.shape[1]+alpha_normal.shape[1]), dtype=np.uint8) * 100
    alpha_up = F.softmax(alpha_up, dim=-1).cpu().detach().numpy()
    alpha_normal = F.softmax(alpha_normal, dim=-1).cpu().detach().numpy()

    for cell_i in range(num_cell):
        for edge_i in range(alpha_up.shape[1]):
            genotype[cell_i][edge_i] = np.argmax(alpha_up[cell_i][edge_i])

        for edge_j in range(alpha_normal.shape[1]):
            genotype[cell_i][offset+edge_j] = np.argmax(alpha_normal[cell_i][edge_j])

    # hard rules
    for cell_i in range(num_cell):
        # node3 of all cell must has a input feature
        if genotype[cell_i][2] + genotype[cell_i][3] == 0:
            edge_j = np.argmin([alpha_normal[cell_i][2 - offset][0], alpha_normal[cell_i][3 - offset][0]])
            genotype[cell_i][offset + edge_j] = np.argmax(np.delete(alpha_normal[cell_i][edge_j], 0)) + 1

        # node4 of all cells must has a input feature
        if genotype[cell_i][4] + genotype[cell_i][5] + genotype[cell_i][6] == 0:
            edge_j = np.argmin([100, 100, alpha_normal[cell_i][2][0], alpha_normal[cell_i][3][0], alpha_normal[cell_i][4][0]])
            genotype[cell_i][offset+edge_j] = np.argmax(np.delete(alpha_normal[cell_i][edge_j], 0)) + 1

    if save:
        np.save(file_path, genotype)

    return genotype


def beta2genotype(beta_normal, beta_down, save=False, file_path=None):
    num_cell = beta_normal.shape[0]
    offset = beta_normal.shape[1]

    genotype = np.ones((beta_normal.shape[0], beta_normal.shape[1]+beta_down.shape[1]), dtype=np.uint8) * 100
    beta_normal = F.softmax(beta_normal, dim=1).cpu().detach().numpy()
    beta_down = F.softmax(beta_down, dim=-1).cpu().detach().numpy()

    for cell_i in range(num_cell):
        for edge_i in range(beta_normal.shape[1]):
            if (cell_i == 0) and (edge_i in [0, 1, 2]):
                genotype[cell_i][edge_i] = np.argmax(np.delete(beta_normal[cell_i][edge_i], [5, 6]))
            else:
                genotype[cell_i][edge_i] = np.argmax(beta_normal[cell_i][edge_i])

        for edge_j in range(beta_down.shape[1]):
            genotype[cell_i][offset+edge_j] = np.argmax(beta_down[cell_i][edge_j])

    # hard rules
    for cell_i in range(1, num_cell):
        # node2 of all cells must have a input feature
        if genotype[cell_i][1] + genotype[cell_i][3] == 0:
            edge_i = np.argmin([100, beta_normal[cell_i][1][0], 100, beta_normal[cell_i][3][0]])
            genotype[cell_i][edge_i] = np.argmax(np.delete(beta_normal[cell_i][edge_i], 0)) + 1

        # node3 of all cells must have a input feature
        if genotype[cell_i][2] + genotype[cell_i][4] == 0:
            edge_i = np.argmin([100, 100, beta_normal[cell_i][2][0], 100, beta_normal[cell_i][4][0]])
            genotype[cell_i][edge_i] = np.argmax(np.delete(beta_normal[cell_i][edge_i], 0)) + 1

        # need node1
        if (genotype[cell_i][3] + genotype[cell_i][4] > 0) and (genotype[cell_i][0] == 0):
            genotype[cell_i][0] = np.argmax(np.delete(beta_normal[cell_i][0], 0)) + 1

    if save:
        np.save(file_path, genotype)

    return genotype


def draw_graph_G(genotype, save=False, file_path=None):
    num_cell, num_edge = genotype.shape[0], genotype.shape[1]
    from graphviz import Digraph
    g = Digraph(
        format='pdf',
        edge_attr=dict(fontsize='20', fontname="times"),
        node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5',
                       penwidth='2', fontname="times"),
        engine='dot')
    g.body.extend(['rankdir=LR'])
    g.node('0', fillcolor='darkseagreen2')

    for i in range(1, 12):
        g.node(str(i), fillcolor='lightblue')
    g.node('12', fillcolor='palegoldenrod')

    for cell_i in range(num_cell):
        ops = []
        for edge_i in range(num_edge):
            if edge_i < 2:
                ops.append(PRIMITIVES_up[genotype[cell_i][edge_i]])
            else:
                ops.append(PRIMITIVES[genotype[cell_i][edge_i]])

        g.edge(str(0+4*cell_i), str(1+4*cell_i), label=ops[0], fillcolor='gray')
        g.edge(str(0+4*cell_i), str(2+4*cell_i), label=ops[1], fillcolor='gray')
        g.edge(str(1+4*cell_i), str(3+4*cell_i), label=ops[2], fillcolor='gray')
        g.edge(str(2+4*cell_i), str(3+4*cell_i), label=ops[3], fillcolor='gray')
        g.edge(str(1+4*cell_i), str(4+4*cell_i), label=ops[4], fillcolor='gray')
        g.edge(str(2+4*cell_i), str(4+4*cell_i), label=ops[5], fillcolor='gray')
        g.edge(str(3+4*cell_i), str(4+4*cell_i), label=ops[6], fillcolor='gray')

    g.edge(str(3), str(7), label='bilinear', fillcolor='gray')
    g.edge(str(3), str(11), label='nearest', fillcolor='gray')
    g.edge(str(7), str(11), label='nearest', fillcolor='gray')

    if save:
        g.render(file_path, view=True)


def draw_graph_D(genotype, save=False, file_path=None):
    num_cell, num_edge = genotype.shape[0], genotype.shape[1]
    from graphviz import Digraph
    g = Digraph(
        format='pdf',
        edge_attr=dict(fontsize='20', fontname="times"),
        node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5',
                       penwidth='2', fontname="times"),
        engine='dot')
    g.body.extend(['rankdir=LR'])
    g.node('0', fillcolor='darkseagreen2')

    for i in range(1, 16):
        g.node(str(i), fillcolor='lightblue')
    g.node('16', fillcolor='palegoldenrod')

    for cell_i in range(num_cell):
        ops = []
        for edge_i in range(num_edge):
            if (cell_i == 0) and (edge_i in [0, 1, 2]):
                ops.append(PRIMITIVES_wo_act[genotype[cell_i][edge_i]])
            elif edge_i < 5:
                ops.append(PRIMITIVES[genotype[cell_i][edge_i]])
            else:
                ops.append(PRIMITIVES_down[genotype[cell_i][edge_i]])

        g.edge(str(0+4*cell_i), str(1+4*cell_i), label=ops[0], fillcolor='gray')
        g.edge(str(0+4*cell_i), str(2+4*cell_i), label=ops[1], fillcolor='gray')
        g.edge(str(0+4*cell_i), str(3+4*cell_i), label=ops[2], fillcolor='gray')
        g.edge(str(1+4*cell_i), str(2+4*cell_i), label=ops[3], fillcolor='gray')
        g.edge(str(1+4*cell_i), str(3+4*cell_i), label=ops[4], fillcolor='gray')
        g.edge(str(2+4*cell_i), str(4+4*cell_i), label=ops[5], fillcolor='gray')
        g.edge(str(3+4*cell_i), str(4+4*cell_i), label=ops[6], fillcolor='gray')

    if save:
        g.render(file_path, view=True)

