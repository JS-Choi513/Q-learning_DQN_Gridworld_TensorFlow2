WHITE_COLOR_RGB = (255,255,255)
BLACK_COLOR_RGB = (0,0,0)
RED_COLOR_RGB = (255, 0, 0)
GREEN_COLOR_RGB = (0, 255, 0)
BLUE_COLOR_RGB = (0, 0, 255)
WIDTH = 500
NUM_ROWS = 5
NUM_EPISODES=30
EPSILON = 4 # Integer 1-10. 2 = 20% random, 3 = 30% random ...
CELL_SIZE = WIDTH/NUM_ROWS
ACTIONS = ['left', 'right', 'up', 'down']
TERMINAL_CELLS = [(2, 2)]
CELL_VALUES = [((1, 1), -10),((3, 3), 10)]
#((6, 1), -10),
#((1, 4), -10),
#((2, 7), -40),
#((2, 8), -10),


