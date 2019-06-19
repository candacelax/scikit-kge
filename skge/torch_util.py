import numpy as np
import functools, collections, torch
from torch import rfft, irfft

def fn_rfft(signal):
    return rfft(signal, signal_ndim=1, onesided=False).transpose(0,1).transpose(0,2)

def fn_irfft(signal):
    return irfft(signal.transpose(0,2).transpose(0,1), signal_ndim=1, onesided=False)

def complex_mult(mat1, mat2): # dimensions (2, m, n) where input[0] is real and input[1] is imaginary
    real1, imag1 = mat1[0], mat1[1]
    real2, imag2 = mat2[0], mat2[1]
    return torch.stack([real1 * real2 - imag1 * imag2, real1 * imag2 + imag1 * real2], dim = -1)\
                .transpose(0,1).transpose(0,2)

def complex_conj(signal): # make imaginary dimension non-negative
    mask = torch.stack((torch.zeros((signal.shape[1], signal.shape[2])),
                        torch.ones((signal.shape[1], signal.shape[2]))), dim=0)
    mask = (signal < 0).float() * mask # only values less than zero
    signal += signal * (-2 * mask)
    return signal

def cconv(a, b):
    """
    Circular convolution of vectors

    Computes the circular convolution of two vectors a and b via their
    fast fourier transforms

    a \ast b = \mathcal{F}^{-1}(\mathcal{F}(a) \odot \mathcal{F}(b))

    Parameter
    ---------
    a: real valued array (shape N)
    b: real valued array (shape N)

    Returns
    -------
    c: real valued array (shape N), representing the circular
       convolution of a and b
    """

    #return fn_irfft( complex_mult( fn_rfft(a), fn_rfft(b) ) )
    a,b = a.detach().numpy(), b.detach().numpy()
    return torch.FloatTensor( np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)).real )

def ccorr(a, b):
    """
    Circular correlation of vectors

    Computes the circular correlation of two vectors a and b via their
    fast fourier transforms

    a \ast b = \mathcal{F}^{-1}(\overline{\mathcal{F}(a)} \odot \mathcal{F}(b))

    Parameter
    ---------
    a: real valued array (shape N)
    b: real valued array (shape N)

    Returns
    -------
    c: real valued array (shape N), representing the circular
       correlation of a and b
    """

    #return fn_irfft( complex_conj( complex_mult( fn_rfft(a), fn_rfft(b) ) ) )
    a,b = a.detach().numpy(), b.detach().numpy()
    return torch.Tensor( np.fft.ifft(np.conj(np.fft.fft(a)) * np.fft.fft(b)).real )


def grad_sum_matrix(idx):
    uidx, iinv = torch.unique(idx, return_inverse=True)
    sz = len(iinv)
    data = torch.ones(sz)
    row_col = torch.stack((iinv, torch.arange(sz)))
    M = torch.sparse.FloatTensor(row_col, data)
    # normalize summation matrix so that each row sums to one
    n = torch.sparse.sum(M, dim=1).values()
    #M = M.T.dot(np.diag(n))
    return uidx, M, n

def computeCosineSimilarity(embeddings):
    dot = embeddings @ torch.t(embeddings)
    norm = embeddings.norm(dim=1).unsqueeze(1)
    cos_sim = dot / torch.mm(norm, torch.t(norm))
    is_nan = len((torch.isnan(cos_sim)).nonzero()) > 0
    assert not is_nan, \
        'Cosine similaritiy issue! embeddings {}\ncos {}'.format(embeddings, cos_sim)
    return cos_sim

''' from most to least similar '''
def getSortedSimilarities(similarities, next_closest_idx=None):
    sorted_similarities, sorted_indices = torch.sort(similarities, dim=1, descending=True)

    if next_closest_idx:
        min_indices = torch.index_select(sorted_indices,
                                         dim=1,
                                         index=torch.LongTensor([next_closest_idx]).to(device))\
                           .squeeze(1).tolist()
        return min_indices
    else:
        return sorted_similarities, sorted_indices.squeeze(1).cpu()

# def init_nvecs(xs, ys, sz, rank, with_T=False):
#     from scipy.sparse.linalg import eigsh

#     T = to_tensor(xs, ys, sz)
#     T = [Tk.tocsr() for Tk in T]
#     S = sum([T[k] + T[k].T for k in range(len(T))])
#     _, E = eigsh(sp.csr_matrix(S), rank)
#     if not with_T:
#         return E
#     else:
#         return E, T


class memoized(object):
    '''
    Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated).

    see https://wiki.python.org/moin/PythonDecoratorLibrary#Memoize
    '''

    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args):
        if not isinstance(args, collections.Hashable):
            # uncachable, return direct function application
            return self.func(*args)
        if args in self.cache:
            return self.cache[args]
        else:
            val = self.func(*args)
            self.cache[args] = val
            return val

    def __repr__(self):
        '''return function's docstring'''
        return self.func.__doc__

    def __get__(self, obj, objtype):
        '''support instance methods'''
        return functools.partial(self.__call__, obj)


class WordEmbeddingLoss:
    # TODO add option to update current "baseline" word embeddings
    # TODO add option to update margins
    def __init__(self, pretrained_embeddings, pos_example_margin, neg_example_margin):
        self._pretrained_embeddings = pretrained_embeddings
        self._pos_example_margin = pos_example_margin
        self._neg_example_margin = neg_example_margin
    ''' 
       pn are differences between pretrained and current embedding
    '''
    def __call__(self, indices, input_embeddings, neighbor_margin=0.6):
        # for pos_neg, 1=positive example, -1=negative example, 0=ignore
        selected_pretrained = self._pretrained_embeddings[indices]
        pretrained_cosine_similarity = computeCosineSimilarity(selected_pretrained)
        pn = (pretrained_cosine_similarity >= neighbor_margin) + (pretrained_cosine_similarity < neighbor_margin)
        
        input_cosine_similarity = computeCosineSimilarity(input_embeddings)

        # loss for positive examples
        delta_from_pretrained = torch.abs(pretrained_cosine_similarity-input_cosine_similarity)
        pos_loss_idxs = (pn==1).float() * (delta_from_pretrained > self._pos_example_margin).float()
        loss_positive_examples = delta_from_pretrained * pos_loss_idxs

        # loss for negative examples
        loss_neg_examples = (pn==-1).float() * torch.max(torch.zeros_like(input_cosine_similarity),
                                                         input_cosine_similarity-self._neg_example_margin)
        loss = (torch.sum(loss_positive_examples) + torch.sum(loss_neg_examples) ) / (indices.shape[0] * indices.shape[0])
        return loss
