import torch


def broadcast_shapes(*shapes):
    if not shapes:
        return torch.Size()
    if len(shapes) == 1 or all(shape == shapes[0] for shape in shapes[1:]):
        return torch.Size(shapes[0])
    if any(type(size) is not int for shape in shapes for size in shape):
        return torch.broadcast_shapes(*shapes)
    result = [1] * max(map(len, shapes))
    for shape in shapes:
        offset = len(result) - len(shape)
        for axis, size in enumerate(shape, offset):
            current = result[axis]
            if current == 1:
                result[axis] = size
            elif size != 1 and size != current:
                raise RuntimeError('Shape mismatch: objects cannot be broadcast to a single shape')
    return torch.Size(result)
