"""
We use sets to store game entities, because we frequently add and remove them.
However, we cannot remove entities from a set when iterating over one, because it ay affect the iteration.
This class stores the instances of set elements to be deleted, and then deletes them.
"""


class LazyRemove:
    """Lazy deleter for sets"""

    element_set: set
    elements_to_remove: list

    def __init__(self, element_set: set) -> None:
        """Instantiate the lazy remove class"""
        self.element_set = element_set
        self.elements_to_remove = []

    def schedule_remove(self, element) -> None:
        """Add an element to the list of elements to be removed"""
        self.elements_to_remove.append(element)

    def apply_remove(self) -> None:
        """Remove all elements that were scheduled for removal"""
        for element in self.elements_to_remove:
            self.element_set.remove(element)
        self.elements_to_remove.clear()

    def clear(self) -> None:
        """Clear the list of elements to be removed"""
        self.elements_to_remove.clear()
