"""Run the game and play it yourself"""

import setup  # pylint: disable=unused-import

from game.frontend.launcher import Launcher

if __name__ == "__main__":

    launcher = Launcher()
    launcher.launch()
