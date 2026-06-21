import jax
from jax.tree_util import register_pytree_node
from jax.typing import ArrayLike


class Key:
    def __init__(self, key: ArrayLike):
        self.__key = jax.random.key(key)

    def __call__(self):
        key, self.__key = jax.random.split(self.__key, 2)
        return key
    
    def _tree_flatten(self):
        children = (self.__key)
        return (children, None)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        obj.__key = children
        return obj


register_pytree_node(Key, Key._tree_flatten, Key._tree_unflatten)
