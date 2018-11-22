
import torch


def pad_tensor(vec, pad, dim):
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.zeros(*pad_size)], dim=dim)


class PadCollate:
    """
    the callback for padding a list of tensors to a common shape in order to be stack together
    """

    def __init__(self, dim=0):
        self.dim = dim

    def pad_collate(self, batch):
        """
        :param batch: a list of tensor/label
        :return:
        """
        # find the longest sequence
        max_len = max(map(lambda x: x[2].shape[self.dim], batch))

        images = []
        logits = []
        bboxes = []
        for i in range(len(batch)):
            curr_boxes = batch[i][2]
            curr_boxes = pad_tensor(curr_boxes, max_len, self.dim)
            images.append(batch[i][0])
            logits.append(batch[i][1])
            bboxes.append(curr_boxes)
        return torch.stack(images, dim=self.dim), \
               torch.stack(logits, dim=self.dim), \
               torch.stack(bboxes, dim=self.dim)

    def __call__(self, batch):
        return self.pad_collate(batch)