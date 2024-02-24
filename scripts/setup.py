"""
Always import this file in python script files from this directory as the first import.
This is needed in order to make python recognize the root "game" module and to enable imports such as

```python
import game.backend
import game.frontend
```
"""

import os
import sys

# Get the absolute path to the parent directory of the game package
game_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the parent directory of the game package to the Python path
sys.path.append(game_parent_dir)
