from Window import Window
from Board import Board
from Player import Player
from QLearner import QLearner
import pygame
import time

pygame.init()
w = Window()
p = Player()
b = Board()
b.createPenaltyCells()
w.drawSurface(b, p)

while True:
    # exit call
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    q = QLearner(b)
    qTable = q.learn()
    dirToGo = {}
    for k, v in qTable.items():
        # Q table iteration
        dirToGo[k] = max(v, key=v.get)
        print("k: {}, v: {}".format(k,v))
    #
    print("____________________________________-")
    for k, v in dirToGo.items():
         print("k: {}, v: {}".format(k,v))

    currNode = (p.getCurrCoords())
    time.sleep(2)
    while(not b.isTerminalCell(currNode)):
        
        time.sleep(2)
        p.move(dirToGo[currNode])
        currNode = p.getCurrCoords()
        # print(currNode)
        w.colorCell(currNode, (0, 0, 255))
        pygame.display.update()
