'''
CSE 691: Topics in Reinforcement Learning Spring 2022, Homework Assignment 3 Implementation
set up a new folder: image_recording under the code path to save the GIF
the implementation is about base policy (Heuristic) and multiagent rollout (1-step lookahead + Heuristic) for spider and
fly within the 10x10 grid world
Lei Zhang
'''

import time
import numpy as np
import os
import pygame as pg
from collections import Counter

class VisUtils:
    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.screen_width = 600
        self.screen_height = 600
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.blockSize = 60

        # fly_pos_show = [[570, 210], [510, 510], [570, 210], [270, 390], [90, 450], [150, 210]]
        self.fly_pos_show = [[8, 2], [8, 8], [4, 6], [1, 7], [2, 3]]
        self.fly_positions = [[8, 2], [8, 8], [4, 6], [1, 7], [2, 3]]
        self.spider_positions_ini = [[6, 0], [6, 0]]

        self.spider1_pos = []
        self.spider2_pos = []

        self.spider1_pos.append(self.spider_positions_ini[0])
        self.spider2_pos.append(self.spider_positions_ini[1])

        pg.init()
        self.viewer = pg.display.set_mode((self.screen_width, self.screen_height))

        self.viewer.fill(self.WHITE)

        for x in range(self.screen_width):
            for y in range(self.screen_height):
                rect = pg.Rect(x * self.blockSize, y * self.blockSize,
                                   self.blockSize, self.blockSize)
                pg.draw.rect(self.viewer, self.BLACK, rect, 1)

        # plot fly positions
        pg.draw.circle(self.viewer, (0, 150, 1),
                       (self.fly_pos_show[0][0] * 60 + 60 // 2, self.fly_pos_show[0][1] * 60 + 60 // 2), 25)
        pg.draw.circle(self.viewer, (0, 150, 1),
                       (self.fly_pos_show[1][0] * 60 + 60 // 2, self.fly_pos_show[1][1] * 60 + 60 // 2), 25)
        pg.draw.circle(self.viewer, (0, 150, 1),
                       (self.fly_pos_show[2][0] * 60 + 60 // 2, self.fly_pos_show[2][1] * 60 + 60 // 2), 25)
        pg.draw.circle(self.viewer, (0, 150, 1),
                       (self.fly_pos_show[3][0] * 60 + 60 // 2, self.fly_pos_show[3][1] * 60 + 60 // 2), 25)
        pg.draw.circle(self.viewer, (0, 150, 1),
                       (self.fly_pos_show[4][0] * 60 + 60 // 2, self.fly_pos_show[4][1] * 60 + 60 // 2), 25)

        # plot spider positions
        pg.draw.circle(self.viewer, (150, 1, 1),
                       (self.spider_positions_ini[0][0] * 60 + 60 // 2, self.spider_positions_ini[0][1] * 60 + 60 // 2), 25)
        pg.draw.circle(self.viewer, (150, 1, 1),
                       (self.spider_positions_ini[1][0] * 60 + 60 // 2, self.spider_positions_ini[1][1] * 60 + 60 // 2), 25)

        if self.algorithm == 'base_policy':
            self.posGenerationHeuristic()

        if self.algorithm == 'multiagent_rollout':
            self.posGenerationMultiAgentRollout()

        self.spider_par = [{'sprite': 'spider.png',   # add this figure later
                            'state': self.spider1_pos},
                           {'sprite': 'spider.png',
                            'state': self.spider2_pos}]

        pg.display.flip()
        pg.display.update()

    '''
    setup grid world and implement animation 
    '''
    def drawFrame(self):
        steps = len(self.spider2_pos)

        for k in range(steps):
            self.viewer.fill(self.WHITE)
            self.drawAxes()
            # Draw Images
            n_spider = 2
            for i in range(n_spider):
                '''getting pos of agent: (x, y)'''
                pos_x, pos_y = np.array(self.spider_par[i]['state'][k])  # spider position
                '''smooth out the movement between each step'''
                if i == 0:
                    pg.draw.circle(self.viewer, (150, 1, 1), (pos_x * 60 + 60 // 2, pos_y * 60 + 60 // 2), 25)
                if i == 1:
                    pg.draw.circle(self.viewer, (150, 1, 1), (pos_x * 60 + 60 // 2, pos_y * 60 + 60 // 2), 25)
                time.sleep(0.05)

            # Annotations
            font = pg.font.SysFont("Arial", 25)
            screen_w, screen_h = self.viewer.get_size()
            label_x = screen_w - 500
            label_y = screen_h - 25
            pg.draw.circle(self.viewer, (0, 150, 1),
                           (label_x, label_y), 15)
            label = font.render("{}".format('fly'), 1, (0, 0, 0))
            self.viewer.blit(label, (label_x + 20, label_y - 15))

            pg.draw.circle(self.viewer, (150, 1, 1),
                           (label_x + 100, label_y), 15)
            label = font.render("{}".format('spider'), 1, (0, 0, 0))
            self.viewer.blit(label, (label_x + 120, label_y - 15))

            recording_path = 'image_recording/'
            pg.image.save(self.viewer, "%simg%03d.png" % (recording_path, k))

            "drawing the map of state distribution"
            pg.display.flip()
            pg.display.update()

    def drawAxes(self):
        # draw grid world and fly positions
        for x in range(self.screen_width):
            for y in range(self.screen_height):
                rect = pg.Rect(x * self.blockSize, y * self.blockSize,
                               self.blockSize, self.blockSize)
                pg.draw.rect(self.viewer, self.BLACK, rect, 1)

        # plot fly positions
        pg.draw.circle(self.viewer, (0, 150, 1),
                       (self.fly_pos_show[0][0] * 60 + 60 // 2, self.fly_pos_show[0][1] * 60 + 60 // 2), 25)
        pg.draw.circle(self.viewer, (0, 150, 1),
                       (self.fly_pos_show[1][0] * 60 + 60 // 2, self.fly_pos_show[1][1] * 60 + 60 // 2), 25)
        pg.draw.circle(self.viewer, (0, 150, 1),
                       (self.fly_pos_show[2][0] * 60 + 60 // 2, self.fly_pos_show[2][1] * 60 + 60 // 2), 25)
        pg.draw.circle(self.viewer, (0, 150, 1),
                       (self.fly_pos_show[3][0] * 60 + 60 // 2, self.fly_pos_show[3][1] * 60 + 60 // 2), 25)
        pg.draw.circle(self.viewer, (0, 150, 1),
                       (self.fly_pos_show[4][0] * 60 + 60 // 2, self.fly_pos_show[4][1] * 60 + 60 // 2), 25)

    '''
    spider position generation through base policy(Heuristic)
    '''
    def posGenerationHeuristic(self):
        for i in range(len(self.fly_positions)):
            # decision making
            idx1 = int(self.spiderChoice(self.spider1_pos[-1])[0])
            idx2 = int(self.spiderChoice(self.spider2_pos[-1])[0])
            fly_pos_observed1 = self.fly_positions[idx1]
            fly_pos_observed2 = self.fly_positions[idx2]
            del self.fly_positions[idx1]
            actions_info_1 = self.actionChoice(self.spider1_pos[-1], fly_pos_observed1)
            actions_info_2 = self.actionChoice(self.spider2_pos[-1], fly_pos_observed2)
            spider1_new_pos = self.spiderPositionUpdate(self.spider1_pos[-1], actions_info_1)
            spider2_new_pos = self.spiderPositionUpdate(self.spider2_pos[-1], actions_info_2)
            for n in range(len(spider1_new_pos)):
                self.spider1_pos.append(spider1_new_pos[n])
                self.spider2_pos.append(spider2_new_pos[n])

    def spiderChoice(self, spider_pos):
        choice = []
        Heuristic = []
        for fly_pos in self.fly_positions:
            Heuristic.append(self.manhattanHeuristic(spider_pos, fly_pos))
        choice.append(np.argmin(Heuristic))
        return choice

    '''
    spider position generation through multiagent rollout(1-step lookahead + Heuristic)
    '''
    def posGenerationMultiAgentRollout(self):
        while True:
            if len(self.fly_positions) == 1:
                pos_lookahead = self.spiderLookahead(self.spider1_pos[-1])
                idx1 = int(self.spiderMultiAgentRolloutChoice(pos_lookahead))
                pos_lookahead = self.spiderLookahead(self.spider2_pos[-1])
                idx2 = int(self.spiderMultiAgentRolloutChoice(pos_lookahead))
                fly_pos_observed1 = self.fly_positions[idx1]
                fly_pos_observed2 = self.fly_positions[idx2]
                del self.fly_positions[idx1]
            else:
                pos_lookahead = self.spiderLookahead(self.spider1_pos[-1])
                idx1 = int(self.spiderMultiAgentRolloutChoice(pos_lookahead))
                fly_pos_observed1 = self.fly_positions[idx1]
                del self.fly_positions[idx1]
                pos_lookahead = self.spiderLookahead(self.spider2_pos[-1])
                idx2 = int(self.spiderMultiAgentRolloutChoice(pos_lookahead))
                fly_pos_observed2 = self.fly_positions[idx2]
                del self.fly_positions[idx2]

            actions_info_1 = self.actionChoice(self.spider1_pos[-1], fly_pos_observed1)
            actions_info_2 = self.actionChoice(self.spider2_pos[-1], fly_pos_observed2)
            spider1_new_pos = self.spiderPositionUpdate(self.spider1_pos[-1], actions_info_1)
            spider2_new_pos = self.spiderPositionUpdate(self.spider2_pos[-1], actions_info_2)

            for n in range(len(spider1_new_pos)):
                self.spider1_pos.append(spider1_new_pos[n])
            for n in range(len(spider2_new_pos)):
                self.spider2_pos.append(spider2_new_pos[n])

            if len(self.fly_positions) == 0:
                break

    def spiderMultiAgentRolloutChoice(self, spider_positions):
        idx = []
        choice = []
        tmp_pos1 = []

        for spider_pos in spider_positions:
            Heuristic = []
            tmp_pos2 = []
            for fly_pos in self.fly_positions:
                Heuristic.append(self.manhattanHeuristic(spider_pos, fly_pos))
                tmp_pos2.append([abs(spider_pos[0] - fly_pos[0]), abs(spider_pos[1] - fly_pos[1])])

            idx.append(np.argmin(Heuristic))
            choice.append(min(Heuristic))
            min_num = Counter(Heuristic)[choice[-1]]

            # when horizontal direction outperform vertical direction in case of tie
            if min_num > 2:
                tmp_pos_x = []
                tmp_pos_idx = []
                min_idx = []
                for e in range(len(Heuristic)):
                    if Heuristic[e] == min(Heuristic):
                        min_idx.append(e)

                idx.pop()
                choice.pop()
                for k in range(len(min_idx)):
                    tmp_pos_idx.append(min_idx[k])
                    tmp_pos_x.append(tmp_pos2[min_idx[k]][0])

                idx.append(np.argmax(tmp_pos_x))
                choice.append(Heuristic[int(np.argmax(tmp_pos_x))])

            tmp_pos1.append([abs(spider_pos[0] - self.fly_positions[int(idx[-1])][0]),
                             abs(spider_pos[1] - self.fly_positions[int(idx[-1])][1])])

        min_num = Counter(choice)[min(choice)]

        if min_num > 2:
            tmp_pos_x = []
            tmp_pos_idx = []
            min_idx = []
            for e in range(len(choice)):
                if choice[e] == min(choice):
                    min_idx.append(e)

            for k in range(len(min_idx)):
                tmp_pos_idx.append(min_idx[k])
                tmp_pos_x.append(tmp_pos1[min_idx[k]][0])

            return idx[int(np.argmax(tmp_pos_x))]

        else:
            return idx[int(np.argmin(choice))]


    '''
    spider position generation through ordinary rollout(1-step lookahead + Heuristic), the simulation result should be 
    similar as the multiagent rollout, but I have no idea to implement this part
    '''

    '''
    utility function package
    '''
    def manhattanHeuristic(self, pos1, pos2):
        "The Manhattan distance heuristic for a PositionSearchProblem"
        xy1 = pos1
        xy2 = pos2
        return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

    def actionChoice(self, pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2
        if x1 > x2:
            if abs(x1 - x2) >= abs(y1 - y2):
                if y1 > y2:
                    actions = (['W'] * (x1 - x2) + ['N'] * (y1 - y2), 'X_first')
                elif y1 < y2:
                    actions = (['W'] * (x1 - x2) + ['S'] * (y2 - y1), 'X_first')
                else:
                    actions = (['W'] * (x1 - x2), 'X_first')
            if abs(x1 - x2) < abs(y1 - y2):
                if y1 > y2:
                    actions = (['N'] * (y1 - y2) + ['W'] * (x1 - x2), 'Y_first')
                elif y1 < y2:
                    actions = (['S'] * (y2 - y1) + ['W'] * (x1 - x2), 'Y_first')
                else:
                    actions = (['W'] * (x1 - x2), 'X_first')
        elif x1 < x2:
            if abs(x1 - x2) >= abs(y1 - y2):
                if y1 > y2:
                    actions = (['E'] * (x2 - x1) + ['N'] * (y1 - y2), 'X_first')
                elif y1 < y2:
                    actions = (['E'] * (x2 - x1) + ['S'] * (y2 - y1), 'X_first')
                else:
                    actions = (['E'] * (x2 - x1), 'X_first')
            if abs(x1 - x2) < abs(y1 - y2):
                if y1 > y2:
                    actions = (['N'] * (y1 - y2) + ['E'] * (x2 - x1), 'Y_first')
                elif y1 < y2:
                    actions = (['S'] * (y2 - y1) + ['E'] * (x2 - x1), 'Y_first')
                else:
                    actions = (['E'] * (x2 - x1), 'X_first')
        else:
            if y1 > y2:
                actions = (['N'] * (y1 - y2), 'Y_first')
            elif y1 < y2:
                actions = (['S'] * (y2 - y1), 'Y_first')

        return actions

    def spiderPositionUpdate(self, spider_pos, actions_info):
        new_pos = []
        count_x = 1
        count_y = 1
        actions, dir = actions_info
        if dir == 'X_first':
            for action in actions:
                if action == 'E':
                    new_pos.append([spider_pos[0] + count_x, spider_pos[1]])
                    count_x += 1
                if action == 'W':
                    new_pos.append([spider_pos[0] - count_x, spider_pos[1]])
                    count_x += 1
                if action == 'N':
                    new_pos.append([new_pos[-1][0], spider_pos[1] - count_y])
                    count_y += 1
                if action == 'S':
                    new_pos.append([new_pos[-1][0], spider_pos[1] + count_y])
                    count_y += 1

        if dir == 'Y_first':
            for action in actions:
                if action == 'E':
                    new_pos.append([spider_pos[0] + count_x, new_pos[-1][1]])
                    count_x += 1
                if action == 'W':
                    new_pos.append([spider_pos[0] - count_x, new_pos[-1][1]])
                    count_x += 1
                if action == 'N':
                    new_pos.append([spider_pos[0], spider_pos[1] - count_y])
                    count_y += 1
                if action == 'S':
                    new_pos.append([spider_pos[0], spider_pos[1] + count_y])
                    count_y += 1

        return new_pos

    def spiderLookahead(self, spider_pos):
        # 1-step lookahead
        pos_x, pos_y = spider_pos
        pos_update = []
        if pos_x == 0:
            action_choice = 4
            steps = [[0, 1], [0, -1], [0, 0], [1, 0]]
            for n in range(action_choice):
                pos_update.append([pos_x + steps[n][0], pos_y + steps[n][1]])
        elif pos_x == 9:
            action_choice = 4
            steps = [[0, 1], [0, -1], [0, 0], [-1, 0]]
            for n in range(action_choice):
                pos_update.append([pos_x + steps[n][0], pos_y + steps[n][1]])
        elif pos_y == 0:
            action_choice = 4
            steps = [[1, 0], [-1, 0], [0, 0], [0, 1]]
            for n in range(action_choice):
                pos_update.append([pos_x + steps[n][0], pos_y + steps[n][1]])
        elif pos_y == 9:
            action_choice = 4
            steps = [[1, 0], [-1, 0], [0, 0], [0, -1]]
            for n in range(action_choice):
                pos_update.append([pos_x + steps[n][0], pos_y + steps[n][1]])
        else:
            action_choice = 5
            steps = [[1, 0], [-1, 0], [0, 1], [0, -1], [0, 0]]
            for n in range(action_choice):
                pos_update.append([pos_x + steps[n][0], pos_y + steps[n][1]])

        return pos_update


if __name__ == '__main__':
    algorithm = ['base_policy', 'multiagent_rollout', 'ordinary_rollout']
    alg_idx = input("Please input your algorithm index:")
    vis = VisUtils(algorithm[int(alg_idx)])
    vis.drawFrame()

    path = 'image_recording/'
    import glob
    image = glob.glob(path+"*.png")
    img_list = image

    import imageio
    images = []
    for filename in img_list:
        images.append(imageio.imread(filename))

    tag = 'spider_fly_' + str(algorithm[int(alg_idx)])
    imageio.mimsave(path + 'movie_' + tag + '.gif', images, 'GIF', duration=0.2)
    # Delete images
    [os.remove(path + file) for file in os.listdir(path) if ".png" in file]
