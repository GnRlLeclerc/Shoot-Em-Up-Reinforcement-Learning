"""Colors enumeration for simplicity"""


class RawEnum:
    """An enum where the values are the keys themselves.
    Allows to access values directly instead of using the .value attribute"""

    def __init__(self, **kwargs) -> None:
        self._values = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getattr__(self, name) -> any:
        return self._values.get(name, None)


class Color(RawEnum):
    """Tuple of RGB colors.
    Access the tuple values directly using Color.<name> instead of Color.<name>.value
    """

    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)

    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)

    YELLOW = (255, 255, 0)
    CYAN = (0, 255, 255)
    MAGENTA = (255, 0, 255)
