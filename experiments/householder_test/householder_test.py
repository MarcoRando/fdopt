import torch
import time
import cProfile as cP


def build_h_spl():
    d = int(1e4)
    l = d

    v = torch.randn(d)
    v.div_(v.norm())

    I = torch.eye(d, l).split(l//2)
#    vs = 
#    print(I[0], v[:l//2].outer(v[:l]))
#    print(I[0].shape)
    H1 = I[0] - 2 * v[:l//2].outer(v[:l])
    H2 = I[1] - 2 * v[l//2:].outer(v[:l])
    H = torch.vstack((H1, H2))
#    print(H.shape)
    return H#torch.vstack((H1, H2))


def build_h():
    d = int(1e4)
    l = d

    v = torch.randn(d)
    v.div_(v.norm())

    I = torch.eye(d, l)
    H = I - 2 * v.outer(v[:l])
 #   print(H.shape)
    return H

it_time = time.time()
build_h()
it_time = time.time() - it_time
print("[--] it time: {}".format(it_time))

it_time = time.time()
Hsp = build_h_spl()
it_time = time.time() - it_time
print("[SPL] it time: {}".format(it_time))


#print(I.split(2, dim =0 ))
#print(I)
#cP.run('build_h()')

#H = - 2 * v.outer(v[:l]) + I
#
#print(v[:2].outer(v[:2]), I[[0, 1]])
#H1 = I[:2, :] - 2 * v.outer(v[:2])
#H2 = I[2:4] - 2 * v[2:4].outer(v[2:4])
#
#print(H)
#print(H1)
#print(H2)
#print(H.T @ H)