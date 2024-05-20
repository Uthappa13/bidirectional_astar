############################################ Import the required libraries ##########################################################
import numpy as np
from queue import PriorityQueue
import time
import cv2
import math as m

################################ Define the lists required for computation of the optimal path #######################################
open1_list = PriorityQueue()
open2_list = PriorityQueue()
close_list = set()

############################################# Define the Configuration Space ########################################################
def config_space():

    # declaring Configuration Space as an array
    c_space = 255 * np.ones((round(3000 / SCALE_FACTOR), round(6000 / SCALE_FACTOR), 3))

    # Boundary + Clearance
    c_space[1:round((clearance + 1) / SCALE_FACTOR), :, :] = BLUE
    c_space[0, :, :] = BLACK

    c_space[round((3000 - clearance - 1) / SCALE_FACTOR):round(2999 / SCALE_FACTOR) - 1, :, :] = BLUE
    c_space[round(2999 / SCALE_FACTOR) - 1, :, :] = BLACK

    c_space[1:round(2999 / SCALE_FACTOR) - 2, 1:round((clearance + 1) / SCALE_FACTOR), :] = BLUE
    c_space[:, 0, :] = BLACK

    c_space[1:round(2999 / SCALE_FACTOR) - 2, round((6000 - clearance - 1) / SCALE_FACTOR):round(5999 / SCALE_FACTOR) - 1, :] = BLUE
    c_space[:, round(5999 / SCALE_FACTOR) - 1, :] = BLACK

    for i in range(1, round(6000 / SCALE_FACTOR) - 1):
        for j in range(1, round(3000 / SCALE_FACTOR) - 1):         
            # Obstacle1
            # Clearance region
            a1 = (i - round(2630 / SCALE_FACTOR))**2 + (j - round(900 / SCALE_FACTOR))**2 - round((700 + clearance) / SCALE_FACTOR)**2
            # Circle region
            b1 = (i - round(2630 / SCALE_FACTOR))**2 + (j - round(900 / SCALE_FACTOR))**2 - round(700 / SCALE_FACTOR)**2
            # initializing pixel values for clearances
            if (a1 < 0):
                c_space[j, i] = BLUE
            # initializing pixel values for obstacles
            if (b1 < 0):
                c_space[j, i] = BLACK

            # Obstacle2
            # Clearance region
            c1 = (i - round(4450 / SCALE_FACTOR))**2 + (j - round(2200 / SCALE_FACTOR))**2 - round((375 + clearance) / SCALE_FACTOR)**2
            # Circle region
            d1 = (i - round(4450 / SCALE_FACTOR))**2 + (j - round(2200 / SCALE_FACTOR))**2 - round(375 / SCALE_FACTOR)**2
            # initializing pixel values for clearances
            if (c1 < 0):
                c_space[j, i] = BLUE
            # initializing pixel values for obstacles
            if (d1 < 0):
                c_space[j, i] = BLACK

            # Obstacle3
            # Clearance region
            k1 = (i - round(1120 / SCALE_FACTOR))**2 + (j - round(2425 / SCALE_FACTOR))**2 - round((400 + clearance ) / SCALE_FACTOR)**2
            # Circle region
            l1 = (i - round(1120 / SCALE_FACTOR))**2 + (j - round(2425 / SCALE_FACTOR))**2 - round(400 / SCALE_FACTOR)**2
            # initializing pixel values for clearances
            if (k1 < 0):
                c_space[j, i] = BLUE
            # initializing pixel values for obstacles
            if (l1 < 0):
                c_space[j, i] = BLACK

    # c_space = cv2.resize(c_space, (600, 200)).astype(np.uint8)
    c_space = cv2.flip(c_space, 0).astype(np.uint8)

    return c_space


############################################################ Action sets ##############################################################
# Defining Actions    
def move_right(current_node, goal, obs_space):
    x = current_node[2][0] + 1
    y = current_node[2][1]

    if np.array_equal(obs_space[y, x, :], BLUE):
        return None
    
    c2c = current_node[1] + 1
    c2g = 1.1 * m.dist((x, y), (goal[0], goal[1]))
    cost = c2c + c2g
    new_node = [cost, c2c, (x, y)]

    return new_node


def move_left(current_node, goal, obs_space):
    x = current_node[2][0] - 1
    y = current_node[2][1]

    if np.array_equal(obs_space[y, x, :], BLUE):
        return None
    
    c2c = current_node[1] + 1
    c2g = 1.1 * m.dist((x, y), (goal[0], goal[1]))
    cost = c2c + c2g
    new_node = [cost, c2c, (x, y)]

    return new_node


def move_up(current_node, goal, obs_space):
    x = current_node[2][0]
    y = current_node[2][1] + 1

    if np.array_equal(obs_space[y, x, :], BLUE):
        return None
    
    c2c = current_node[1] + 1
    c2g = 1.1 * m.dist((x, y), (goal[0], goal[1]))
    cost = c2c + c2g
    new_node = [cost, c2c, (x, y)]

    return new_node


def move_down(current_node, goal, obs_space):
    x = current_node[2][0]
    y = current_node[2][1] - 1

    if np.array_equal(obs_space[y, x, :], BLUE):
        return None
    
    c2c = current_node[1] + 1
    c2g = 1.1 * m.dist((x, y), (goal[0], goal[1]))
    cost = c2c + c2g
    new_node = [cost, c2c, (x, y)]

    return new_node


def move_up_right(current_node, goal, obs_space):
    x = current_node[2][0] + 1
    y = current_node[2][1] + 1

    if np.array_equal(obs_space[y, x, :], BLUE):
        return None
    
    c2c = current_node[1] + 1.4
    c2g = 1.1 * m.dist((x, y), (goal[0], goal[1]))
    cost = c2c + c2g
    new_node = [cost, c2c, (x, y)]

    return new_node


def move_up_left(current_node, goal, obs_space):
    x = current_node[2][0] - 1
    y = current_node[2][1] + 1

    if np.array_equal(obs_space[y, x, :], BLUE):
        return None
    
    c2c = current_node[1] + 1.4
    c2g = 1.1 * m.dist((x, y), (goal[0], goal[1]))
    cost = c2c + c2g
    new_node = [cost, c2c, (x, y)]

    return new_node


def move_down_right(current_node, goal, obs_space):
    x = current_node[2][0] + 1
    y = current_node[2][1] - 1

    if np.array_equal(obs_space[y, x, :], BLUE):
        return None
    
    c2c = current_node[1] + 1.4
    c2g = 1.1 * m.dist((x, y), (goal[0], goal[1]))
    cost = c2c + c2g
    new_node = [cost, c2c, (x, y)]

    return new_node


def move_down_left(current_node, goal, obs_space):
    x = current_node[2][0] - 1
    y = current_node[2][1] - 1

    if np.array_equal(obs_space[y, x, :], BLUE):
        return None
    
    c2c = current_node[1] + 1.4
    c2g = 1.1 * m.dist((x, y), (goal[0], goal[1]))
    cost = c2c + c2g
    new_node = [cost, c2c, (x, y)]

    return new_node



#################################################### Defining Proximity Check function ##################################################
def is_close(node, close_list):
    for closed_node in close_list:
        if np.linalg.norm(np.array(node[:2]) - np.array(closed_node[:2])) < 5:
            return True
    return False


########################################## Implementation of the bi-directional A star algorithm ############################################
def bi_astar(start, goal):
    # structure of node: (cost_to_come, (x cordinate, y cordinate))
    start_cost = round(np.sqrt((start[0] - goal[0])**2 + (start[1] - goal[1])**2))
    start1_node = (start_cost, 0, start)
    start2_node = (start_cost, 0, goal)
    open1_list.put(start1_node)
    open2_list.put(start2_node)
    # creating a dict to track parent child relations for backtracking
    parent1_map = {}
    parent2_map = {}

    path1 = []
    path2 = []

    parent1_map[start] = None
    parent2_map[goal] = None

    current1_node = start1_node
    current2_node = start2_node

    count = 0

    while (not open1_list.empty()) and (not open2_list.empty()):
        if count % 2 == 0:
            count += 1
            current1_node = open1_list.get()

            if current1_node[2] in close_list:
                continue

            close_list.add(current1_node[2])

            if m.dist((current1_node[2][0], current1_node[2][1]), (current2_node[2][0], current2_node[2][1])) <= (20 / SCALE_FACTOR):
                print('Goal Reached!')
                # calling backtracking function after goal node is reached
                path1 = back_tracking(parent1_map, start, current1_node[2])
                path2 = back_tracking(parent2_map, goal, current2_node[2])
                return path1, path2, parent1_map, parent2_map
            
            next_nodes = []

            action_node1 = move_up(current1_node, goal, obs_space)
            if action_node1 != None:
                next_nodes.append(action_node1)

            action_node2 = move_down(current1_node, goal, obs_space)
            if action_node2 != None:
                next_nodes.append(action_node2)

            action_node3 = move_right(current1_node, goal, obs_space)
            if action_node3 != None:
                next_nodes.append(action_node3)

            action_node4 = move_left(current1_node, goal, obs_space)
            if action_node4 != None:
                next_nodes.append(action_node4)

            action_node5 = move_up_right(current1_node, goal, obs_space)
            if action_node5 != None:
                next_nodes.append(action_node5)

            action_node6 = move_up_left(current1_node, goal, obs_space)
            if action_node6 != None:
                next_nodes.append(action_node6)

            action_node7 = move_down_right(current1_node, goal, obs_space)
            if action_node7 != None:
                next_nodes.append(action_node7)

            action_node8 = move_down_left(current1_node, goal, obs_space)
            if action_node8 != None:
                next_nodes.append(action_node8)

            for next_node in next_nodes:
                if next_node[2] not in close_list:
                    if next_node[2] not in [x[2] for x in open1_list.queue]:
                        parent1_map[next_node[2]] = current1_node[2]
                        open1_list.put(tuple(next_node))
                    
                    else:
                        for node in open1_list.queue:
                            if node[2] == next_node[2] and node[0] > next_node[0]:
                                open1_list.queue.remove(node)
                                parent1_map[next_node[2]] = current1_node[2]
                                open1_list.put(tuple(next_node))
                  
        else:
            count += 1
            current2_node = open2_list.get()

            if current2_node[2] in close_list:
                continue

            close_list.add(current2_node[2])

            if m.dist((current1_node[2][0], current1_node[2][1]), (current2_node[2][0], current2_node[2][1])) <= (20 / SCALE_FACTOR):
                print('Goal Reached!')
                # calling backtracking function after goal node is reached
                path1 = back_tracking(parent1_map, start, current1_node[2])
                path2 = back_tracking(parent2_map, goal, current2_node[2])
                return path1, path2, parent1_map, parent2_map
            
            next_nodes = []

            action_node1 = move_up(current2_node, start, obs_space)
            if action_node1 != None:
                next_nodes.append(action_node1)

            action_node2 = move_down(current2_node, start, obs_space)
            if action_node2 != None:
                next_nodes.append(action_node2)

            action_node3 = move_right(current2_node, start, obs_space)
            if action_node3 != None:
                next_nodes.append(action_node3)

            action_node4 = move_left(current2_node, start, obs_space)
            if action_node4 != None:
                next_nodes.append(action_node4)

            action_node5 = move_up_right(current2_node, start, obs_space)
            if action_node5 != None:
                next_nodes.append(action_node5)

            action_node6 = move_up_left(current2_node, start, obs_space)
            if action_node6 != None:
                next_nodes.append(action_node6)

            action_node7 = move_down_right(current2_node, start, obs_space)
            if action_node7 != None:
                next_nodes.append(action_node7)

            action_node8 = move_down_left(current2_node, start, obs_space)
            if action_node8 != None:
                next_nodes.append(action_node8)

            for next_node in next_nodes:
                if next_node[2] not in close_list:
                    if next_node[2] not in [x[2] for x in open2_list.queue]:
                        parent2_map[next_node[2]] = current2_node[2]
                        open2_list.put(tuple(next_node))
                    
                    else:
                        for node in open2_list.queue:
                            if node[2] == next_node[2] and node[0] > next_node[0]:
                                open2_list.queue.remove(node)
                                parent2_map[next_node[2]] = current2_node[2]
                                open2_list.put(tuple(next_node))
    
    
    else:
        print("Goal could not be reached!")
        exit()

########################################## Backtracking function based on parent map ##################################################
def back_tracking(parent_map, start, goal):
    path = []
    current_node = goal

    while current_node != start:
        path.append(current_node)
        current_node = parent_map[current_node]

    path.append(start)
    path.reverse()

    return path

# ########################################################## Main function  #############################################################

if __name__ == '__main__':

    clearance = 20
    SCALE_FACTOR = 10

    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)

    # creating configuration space
    obs_space = config_space()
    # Define start and goal points as per map
    start_point = (int(clearance / SCALE_FACTOR), 150) 
    goal_point = (int(600 - clearance / SCALE_FACTOR), 150)

    # timer object to measure computation time
    timer_start = time.time()

    # implementing dijkstra
    optimal_path1, optimal_path2, visit_map1, visit_map2 = bi_astar(start_point, goal_point)

    # creating a visualization window
    cv2.namedWindow('Optimal Path Animation', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Optimal Path Animation', round(6000 / SCALE_FACTOR), round(4000 / SCALE_FACTOR))

    canvas = obs_space

    timer_stop = time.time()
    c_time = timer_stop - timer_start
    print("Total Runtime: ", c_time)


    ############################################## Display the node exploration #######################################################
    # displaying node exploration
    for key in visit_map1.keys():
        if visit_map1[key] == None:
            continue

        adjusted_parent_point = (int(visit_map1[key][0]), int(visit_map1[key][1]))
        adjusted_child_point = (int(key[0]), int(key[1]))

        cv2.arrowedLine(canvas, pt1=adjusted_parent_point, pt2=adjusted_child_point, color=(0, 255, 0), thickness=1, tipLength=0.2)

    for key in visit_map2.keys():
        if visit_map2[key] == None:
            continue

        adjusted_parent_point = (int(visit_map2[key][0]), int(visit_map2[key][1]))
        adjusted_child_point = (int(key[0]), int(key[1]))

        cv2.arrowedLine(canvas, pt1=adjusted_parent_point, pt2=adjusted_child_point, color=(0, 255, 0), thickness=1, tipLength=0.2)

    # displaying optimal path
    for point in optimal_path1:
        adjusted_point = (int(point[0]), int(point[1]))

        cv2.circle(canvas, adjusted_point, 2, (0, 0, 255), -1)

    for point in optimal_path2:
        adjusted_point = (int(point[0]), int(point[1]))

        cv2.circle(canvas, adjusted_point, 2, (0, 0, 255), -1)

    cv2.imshow('Optimal Path Animation', canvas)

    # holding final frame till any key is pressed
    cv2.waitKey(0)
    # destroying visualization window after keypress
    cv2.destroyAllWindows()





