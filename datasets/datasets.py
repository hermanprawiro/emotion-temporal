def define_dataset(**kwargs):
    dataset = kwargs['dataset']
    sampling = kwargs['sampling']

    if dataset == 'ctbc2':
        from datasets.ctbc_v2 import CTBC_Sparse, CTBC_Dense
        if sampling == 'sparse':
            return CTBC_Sparse
        elif sampling == 'dense':
            return CTBC_Dense
        else:
            raise NotImplementedError('CTBC v2 only supports Sparse and Dense')
    elif dataset == 'enterface':
        from datasets.enterface import Enterface_Sparse, Enterface_Dense
        if sampling == 'sparse':
            return Enterface_Sparse
        elif sampling == 'dense':
            return Enterface_Dense
        else:
            raise NotImplementedError('Enterface only supports Sparse and Dense')
    else:
        raise NotImplementedError('Dataset not yet implemented')