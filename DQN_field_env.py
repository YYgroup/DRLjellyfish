"""
流场环境,调用IBAMR由当前时间步求解下一个时间步
由下一时间步的数据计算得出下一时间步物体的状态
"""
import os
from turtle import pos
import numpy as np
from typing import Callable, Tuple
from DQN_utils import calculate_reward_from_raw_state, transform_raw_state
import torch
from cmath import pi
import random

#读文件名
with open('input2d','r') as f:
    lines = f.readlines()
    for line in lines:
        if 'structure_names' in line:
            filename = line.split()[-1][1:-1]

#读beam的masterPoint的编号顺序
beam_index=[]
with open(filename+".beam",'r') as f:
    num=int(f.readline())
    for i in range(num):
        line=f.readline()
        beam_index.append(int(line.split()[1]))
        k_max = float(line.split()[3]) # k_beam最大值
step_size = 250 #IBAMR中单步的间隔
with open('input2d', 'r') as f:
    lines = f.readlines()
    for line in lines:
        tmp = line.split()
        if len(tmp) > 0 and tmp[0] == 'DUMP':
            step_size = int(tmp[2])
            break    

action_space = []
big = 0.003 
small = 0.001 
temp = [[big, big], [small, big], [big, small], [0, 0]]
for i in temp:
    action_space.append(torch.tensor(i, dtype=torch.float64))
class gridInfo:  
    def __init__(self):  
        self.i = -1
        self.j = -1
        self.level = -1
        self.parent = [-1,-1]
class Flow_Field:
    def __init__(self):
        self.startTime = 0  # 模拟起始时间
        self.endTime = self.read_end_time()  # 终止时间
        self.dt = self.read_dt()  # 时间间隔
        self.currstep = 0
        
        self.currenTimeStep = 0  # 当前时间步，第0步为t=0
        self.currentTime = 0  # 当前时间
        self.Lagpoints = self.read_Lagpoint(self.currenTimeStep)  # 读入当前时间步的拉格朗日点和力
        self.massCenter = self.calculate_mass_center(self.Lagpoints)  # 计算质心坐标
        self.pre_location = self.massCenter

        self.Force = self.read_force(self.currenTimeStep)  # 计算拉格朗日点受力受力
        self.torque = self.calculate_torque(self.Lagpoints, self.massCenter, self.Force)  # 计算对质心的力矩

        self.start_point = self.massCenter.copy()

        # need to call initialize function
        self.targetPoint = np.array([0.8, 0.8])
        self.pre_targetPoint = self.targetPoint.copy()
        self.initTarget = self.targetPoint.copy()
        self.targetList = []

        self.target_vel = np.array([0.0, 0.0])

        self.massCenterList = []
        self.massCenterList.append(self.massCenter)

        self.new_obs = np.zeros(22)
        self.action_space = action_space

        self.target_traj = None
        self.ref_orig = np.array([0.1,0.1])
        self.grid_wid = 0.2

        self.unknownWallPoints = self.readWallPoints()  # wall points set
        self.knownWallPoints = set()

        self.senseRadius = 0.4
        self.roadWidth = 0.1
        self.deltaRotate = pi/18
        self.lastRotate_plus = 0
        self.lastRotate_minus = 0
        self.direction=""

    # 从输入文件读入终止时间
    @staticmethod
    def read_end_time():
        with open('input2d', 'r') as f:
            lines = f.readlines()
        for line in lines:
            tmp = line.split()
            if len(tmp) != 0 and tmp[0] == 'END_TIME':
                return float(tmp[2])

    # 读入dt
    @staticmethod
    def read_dt():
        with open('input2d', 'r') as f:
            lines = f.readlines()
        dump, dt = 0, 0
        for line in lines:
            tmp = line.split()
            if len(tmp) > 0:
                if tmp[0] == 'DT':
                    dt = float(tmp[2])
                elif tmp[0] == 'DUMP':
                    dump = float(tmp[2])
                    #step_size = int(dump)
                    break
        return dump * dt

   
    @staticmethod
    def rearrange_new(data):
        n=len(data)
        gg = data.copy()
        for i in range(41):
            temp = gg[i].copy()
            gg[i] = gg[79-i].copy()
            gg[79-i] = temp.copy()
        return gg

    @staticmethod
    def readWallPoints():
        with open(filename+'.vertex', 'r') as f:
            lag = np.loadtxt(f.name, unpack=True, skiprows=1)
        wall = lag.T[159:]
        wall = np.array(wall)
        [n, _] = wall.shape
        wallPoints = set()
        for i in range(n):
            wallPoints.add(tuple(wall[i,:]))
        return wallPoints


    def isThereTruelyWallBetweenTwoPoints(self, point1, point2, all_wall_pointes):
        x1 = point1[0]
        y1 = point1[1]
        x2 = point2[0]
        y2 = point2[1]
        for item in all_wall_pointes:
            x = item[0]
            y = item[1]
            if ((y2-y1)*y+(x2-x1)*x-(x2**2-x1**2)/2-(y2**2-y1**2)/2)**2/(((x1-x2)**2+(y1-y2)**2)/4) + (-(y2-y1)*x+(x2-x1)*y+(x1*y2-x2*y1))**2/self.roadWidth**2 <= (x1-x2)**2+(y1-y2)**2:
                return True
        return False

    def isThereWallBetweenTwoPoints(self,mass, target):
        # mass:(x2,y2)   target:(x1,y1)
        A = (mass[1]-target[1]) #y2-y1
        B = -(mass[0]-target[0]) #-x2+x1
        C = mass[0]*target[1] - mass[1]*target[0] #x2*y1-x1*y2
        pm = self.roadWidth*(A**2+B**2)**0.5
        x1 = target[0]
        y1 = target[1]
        x2 = mass[0]
        y2 = mass[1]
        distance = np.linalg.norm(target-mass)
        roadWidth = self.roadWidth
        for item in self.knownWallPoints:
            x = item[0]
            y = item[1]
            #if (A*item[0]+B*item[1]+C-pm)*(A*item[0]+B*item[1]+C+pm)<0 and (-B*item[0]+A*item[1]-A*self.massCenter[1]+B*self.massCenter[0])*(-B*item[0]+A*item[1]-A*target[1]+B*target[0])<0:
            #if ((y2-y1)*y+(x2-x1)*x-(x2**2-x1**2)/2-(y2**2-y1**2)/2)**2/(((x1-x2)**2+(y1-y2)**2)/4) + (-(y2-y1)*x+(x2-x1)*y+(x1*y2-x2*y1))**2/roadWidth**2 <= (x1-x2)**2+(y1-y2)**2:
            if ((y2-y1)*y+(x2-x1)*x-(x2**2-x1**2)/2-(y2**2-y1**2)/2)**2/(((x1-x2)**2+(y1-y2)**2)/4) + (-(y2-y1)*x+(x2-x1)*y+(x1*y2-x2*y1))**2/roadWidth**2 <= (x1-x2)**2+(y1-y2)**2:
                print(item,"in the way!")
                return True
            if (A*x+B*y+C-pm)*(A*x+B*y+C+pm)<0 and (A*y-B*x-(x2**2-x1**2)/2-(y2**2-y1**2)/2)*(-B*x+A*y-A*y1+B*x1)<0:
                print(item,"in the way!")
                return True
        return False
    
    def updateKnownWallPointsBasedOnPoint(self, mass):
        for item in self.unknownWallPoints:
            distance = np.linalg.norm(mass-item)
            if distance<self.senseRadius:
                self.knownWallPoints.add(item)
        print("known points:", len(self.knownWallPoints))
        self.unknownWallPoints = self.unknownWallPoints - self.knownWallPoints

    def evaluateDifficulty(self):
        difficulty=0
        vertual_mass = self.massCenter
        vertual_target = self.initTarget
        last_turn = ""
        last_plus_turn = 0
        last_minus_turn = 0
        f = open("difficulty_vertual_traj.txt","w")
        f.write(str(vertual_mass[0]) + ' ' + str(vertual_mass[1]) + '\n')
        while np.linalg.norm(vertual_mass-self.initTarget)>0.03:
            self.updateKnownWallPointsBasedOnPoint(vertual_mass)
            vertual_target = self.initTarget
            distance = np.linalg.norm(vertual_target-vertual_mass)
            vertual_target = vertual_mass + (vertual_target-vertual_mass)/distance*min(0.4,distance)

            if not self.isThereWallBetweenTwoPoints(vertual_mass, vertual_target):
                vertual_mass = vertual_target
                f.write(str(vertual_mass[0]) + ' ' + str(vertual_mass[1]) + '\n')
                difficulty+=1
                continue
            
            count_plus = 0
            vertual_target_plus = vertual_target
            while self.isThereWallBetweenTwoPoints(vertual_mass, vertual_target_plus) and count_plus<pi/self.deltaRotate:
                delta = vertual_target_plus - vertual_mass
                vertual_target_plus = vertual_mass + np.array([delta[0]*np.cos(self.deltaRotate)-delta[1]*np.sin(self.deltaRotate), delta[1]*np.cos(self.deltaRotate)+delta[0]*np.sin(self.deltaRotate)])
                count_plus+=1
           
            count_minus = 0
            vertual_target_minus = vertual_target
            while self.isThereWallBetweenTwoPoints(vertual_mass, vertual_target_minus) and count_minus<pi/self.deltaRotate:
                delta = vertual_target_minus - vertual_mass
                vertual_target_minus = vertual_mass + np.array([delta[0]*np.cos(-self.deltaRotate)-delta[1]*np.sin(-self.deltaRotate), delta[1]*np.cos(-self.deltaRotate)+delta[0]*np.sin(-self.deltaRotate)])
                count_minus+=1
            #print(vertual_mass)
            if last_turn=="":
                if count_plus<count_minus:
                    vertual_mass = vertual_target_plus
                    last_turn="plus"
                elif count_plus>count_minus:
                    vertual_mass = vertual_target_minus
                    last_turn="minus"
                elif count_plus==count_minus:
                    if random.random()<0.5:
                        vertual_mass = vertual_target_minus
                        last_turn="minus"
                    else:
                        vertual_mass = vertual_target_plus
                        last_turn="plus"
            elif last_turn=="minus":
                if abs(last_minus_turn-count_minus)<=last_minus_turn+count_plus:
                    vertual_mass = vertual_target_minus
                    last_turn="minus"
                else:
                    vertual_mass = vertual_target_plus
                    last_turn="plus"
            elif last_turn=="plus":   
                if abs(last_plus_turn-count_plus)<=last_plus_turn+count_minus:
                    vertual_mass = vertual_target_plus
                    last_turn="plus"
                else:
                    vertual_mass = vertual_target_minus
                    last_turn="minus"
            f.write(str(vertual_mass[0]) + ' ' + str(vertual_mass[1]) + '\n')
            last_minus_turn = count_minus
            last_plus_turn = count_plus
            difficulty+=1
        f.write('difficulty: '+str(difficulty))
        f.close()    
        return difficulty




    def updateKnownWallPoints(self):
        for item in self.unknownWallPoints:
            distance = np.linalg.norm(self.massCenter-item)
            if distance<self.senseRadius:
                self.knownWallPoints.add(item)
        print("known points:", len(self.knownWallPoints))
        self.unknownWallPoints = self.unknownWallPoints - self.knownWallPoints

    def isThereWallBetweenMassAndTarget(self, target=None):
        if target is None:
            target = self.initTarget
        # mass:(x2,y2)   target:(x1,y1)
        A = (self.massCenter[1]-target[1]) #y2-y1
        B = -(self.massCenter[0]-target[0]) #-x2+x1
        C = self.massCenter[0]*target[1] - self.massCenter[1]*target[0] #x2*y1-x1*y2
        pm = self.roadWidth*(A**2+B**2)**0.5
        x1 = target[0]
        y1 = target[1]
        x2 = self.massCenter[0]
        y2 = self.massCenter[1]
        distance = np.linalg.norm(target-self.massCenter)
        roadWidth = self.roadWidth
        for item in self.knownWallPoints:
            x = item[0]
            y = item[1]
            #if (A*item[0]+B*item[1]+C-pm)*(A*item[0]+B*item[1]+C+pm)<0 and (-B*item[0]+A*item[1]-A*self.massCenter[1]+B*self.massCenter[0])*(-B*item[0]+A*item[1]-A*target[1]+B*target[0])<0:
            #if ((y2-y1)*y+(x2-x1)*x-(x2**2-x1**2)/2-(y2**2-y1**2)/2)**2/(((x1-x2)**2+(y1-y2)**2)/4) + (-(y2-y1)*x+(x2-x1)*y+(x1*y2-x2*y1))**2/roadWidth**2 <= (x1-x2)**2+(y1-y2)**2:
            if ((y2-y1)*y+(x2-x1)*x-(x2**2-x1**2)/2-(y2**2-y1**2)/2)**2/(((x1-x2)**2+(y1-y2)**2)/4) + (-(y2-y1)*x+(x2-x1)*y+(x1*y2-x2*y1))**2/roadWidth**2 <= (x1-x2)**2+(y1-y2)**2:
                print(item,"in the way!")
                return True
            if (A*x+B*y+C-pm)*(A*x+B*y+C+pm)<0 and (A*y-B*x-(x2**2-x1**2)/2-(y2**2-y1**2)/2)*(-B*x+A*y-A*y1+B*x1)<0:
                print(item,"in the way!")
                return True
        return False


    def rotate(self,tempTarget, angle):
        [x,y]=tempTarget-self.massCenter
        delta = np.array([x*np.cos(angle)-y*np.sin(angle), y*np.cos(angle)+x*np.sin(angle)])
        return self.massCenter+delta


    def rotateFromInitTargetWithinRadius(self):
        distance = np.linalg.norm(self.initTarget-self.massCenter)
        tempTarget = self.massCenter + (self.initTarget-self.massCenter)/distance*min(0.4,distance)
        old_target = self.targetPoint
        self.targetPoint = tempTarget
        print("current target", tempTarget)

        if not self.isThereWallBetweenMassAndTarget(target=tempTarget):
            self.lastRotate_minus=0
            self.lastRotate_plus=0
            self.direction=""
            return
        count_plus = 0
        newTar_plus = tempTarget.copy()
        while self.isThereWallBetweenMassAndTarget(target=newTar_plus) and count_plus<pi/self.deltaRotate:
            newTar_plus = self.rotate(newTar_plus, self.deltaRotate)
            count_plus+=1
           
        count_minus = 0
        newTar_minus = tempTarget.copy() 
        while self.isThereWallBetweenMassAndTarget(target=newTar_minus) and count_minus<pi/self.deltaRotate:
            newTar_minus = self.rotate(newTar_minus, -self.deltaRotate)
            count_minus+=1
        #print("current target", tempTarget)
        if self.lastRotate_plus==0 and self.lastRotate_minus==0: #上一次是直走的,这一次不能直走
            if count_plus<count_minus:
                self.targetPoint = newTar_plus
                self.direction="plus"
            elif count_plus>count_minus:
                self.targetPoint = newTar_minus
                self.direction="minus"
            elif count_plus==count_minus:
                if random.random()<0.5:
                    self.targetPoint = newTar_plus
                    self.direction="plus"
                else:
                    self.targetPoint = newTar_minus
                    self.direction="minus"
        else:
            if self.direction=="plus":
                if abs(count_plus-self.lastRotate_plus)<=self.lastRotate_plus+count_minus:
                    if abs(count_plus-self.lastRotate_plus)>9:
                        self.targetPoint=old_target
                        return
                    else:
                        self.targetPoint = newTar_plus
                        self.direction="plus"
                else:
                    if self.lastRotate_plus+count_minus>9:
                        self.targetPoint=old_target
                        return
                    else:
                        self.targetPoint = newTar_minus
                        self.direction="minus"
            elif self.direction=="minus":
                if abs(count_minus-self.lastRotate_minus)<=self.lastRotate_minus+count_plus:
                    if abs(count_minus-self.lastRotate_minus)>9:
                        self.targetPoint=old_target
                        return
                    else:
                        self.targetPoint = newTar_minus
                        self.direction="minus"
                else:
                    if self.lastRotate_minus+count_plus>9:
                        self.targetPoint=old_target
                        return
                    else:
                        self.targetPoint = newTar_plus
                        self.direction="plus"
        self.lastRotate_plus = count_plus
        self.lastRotate_minus = count_minus
        """
        delta_theta_plus = np.arcsin(dis_plus/2/min(0.4,distance))
        delta_theta_minus = np.arcsin(dis_minus/2/min(0.4,distance))

        if dis_plus < dis_minus and delta_theta_plus<pi/4:
            self.targetPoint = newTar_plus
        elif dis_plus > dis_minus and delta_theta_minus<pi/4:
            self.targetPoint = newTar_minus
        else:
            tempTarget = self.massCenter + (self.initTarget-self.massCenter)/distance*min(0.3,distance)
            count_plus = 0
            newTar_plus = tempTarget.copy()
            while self.isThereWallBetweenMassAndTarget(newTar_plus) and count_plus<pi/self.deltaRotate:
                newTar_plus = self.rotate(newTar_plus, self.deltaRotate)
                count_plus+=1
            count_minus = 0
            newTar_minus = tempTarget.copy() 
            while self.isThereWallBetweenMassAndTarget(newTar_minus) and count_minus<pi/self.deltaRotate:
                newTar_minus = self.rotate(newTar_minus, -self.deltaRotate)
                count_minus+=1
            
            dis_plus = np.linalg.norm(old_target-newTar_minus)
            dis_minus = np.linalg.norm(old_target-newTar_plus)

            if dis_plus < dis_minus:
                self.targetPoint = newTar_plus
            elif dis_plus > dis_minus:
                self.targetPoint = newTar_minus
        """

    def SenseSurroundingAreaAndChangeCourse(self):
        self.updateKnownWallPoints()
        #needToRotate = self.isThereWallBetweenMassAndTarget(self.initTarget)
        #if needToRotate:
        self.rotateFromInitTargetWithinRadius()
        #else:
        #    self.targetPoint = self.initTarget


    # 读入当前时间步的拉格朗日点集合和受力
    @staticmethod
    def read_Lagpoint(timestep:int):
        # t=0初始化
        if timestep == 0:
            with open(filename+'.vertex', 'r') as f:
                lag = np.loadtxt(f.name, unpack=True, skiprows=1)
            lag = lag.T[:159]
        else:
            lag = []
            tag = str(timestep)
            while len(tag) < 5:
                tag = '0' + tag
            lagPointer = "X."+tag
            print('============== reading ' + lagPointer + '================\n')
            with open(os.path.join('hier_data_IB2d', lagPointer), "r") as f:
                for _ in range(3):
                    f.readline()
                lines = f.readlines()
                n = len(lines)//2
                n = 159
                for i in range(n):
                    lag.append([float(lines[2*i]),float(lines[2*i+1])])                
            print('============= read in lagrange points number: ' + str(len(lag)) + '\n')
        return np.array(lag)

    # 计算当前物体质心
    @staticmethod
    def calculate_mass_center(lag):
        return np.mean(lag, axis=0)

    # 计算对心力矩
    @staticmethod
    def calculate_torque(lag, center, Force):
        fx=[]
        fy=[]
        for element in Force:
            fx.append(element[0])
            fy.append(element[1])
        fx=np.array(fx)
        fy=np.array(fy)
        tmp = (lag - center) * np.stack([fy, -fx], axis=1)
        return np.sum(tmp)

    # 读入拉格朗日点上力分布
    @staticmethod
    def read_force(timestep:int):
         # t=0初始化
        if timestep == 0:
            with open(filename+'.vertex', 'r') as f:
                lag = np.loadtxt(f.name, unpack=True, skiprows=1)
            lag = lag.T[:159]
            lag = lag * 0
        else:
            lag = []
            tag = str(timestep)
            while len(tag) < 5:
                tag = '0' + tag
            lagPointer = "F."+tag
            print('============== reading ' + lagPointer + '================\n')
            with open(os.path.join('hier_data_IB2d', lagPointer), "r") as f:
                for _ in range(3):
                    f.readline()
                lines = f.readlines()
                n = len(lines)//2
                n = 159
                for i in range(n):
                    lag.append([float(lines[2*i]),float(lines[2*i+1])])                
            print('============= read in lagrange forces number: ' + str(len(lag)) + '\n')
        return np.array(lag)
    
    @staticmethod
    def update_file_with_line_func(filepath: str, line_func: Callable[[str], Tuple[bool, str]]):
        with open(filepath, 'r') as f:
            lines = f.readlines()
        iter = (line_func(lines) for lines in lines)
        iter = filter(lambda x: x[0], iter)
        iter = (x[1] for x in iter)
        with open(filepath, "w+") as f:
            f.writelines(iter)
  
    def readMazeAndPlanPath(self, mazefile="mazeinfo.txt", target = np.array([0, 0])):
        #print("load in maze info file",mazefile)
        maze = np.loadtxt(mazefile, dtype=int)
        #print("load in maze info file",mazefile)
        [startX, startY] = np.floor((self.massCenter - self.ref_orig)/self.grid_wid).astype(int)
        [endX, endY] = np.floor((target - self.ref_orig)/self.grid_wid).astype(int)
        print("start block at ({}, {}), target block at ({}, {})".format(startX,startY,endX,endY))
        # wall 0, road 1
        if maze[startX-1,startY-1]==0 or maze[endX-1,endY-1]==0:
            print("stuck!")
            return
        [nx, ny] = maze.shape
        info = [[gridInfo() for _ in range(ny)] for _ in range(nx)] 
        for i in range(nx):
            for j in range(ny):
                info[i][j].i = i
                info[i][j].j = j
        iterList=[info[startX-1][startY-1]]
        currentLevel = -1
        while info[endX-1][endY-1].level == -1:
            currentLevel=currentLevel+1
            currentLength=len(iterList)
            for i in range(currentLength):
                pointer=iterList[0]
                del iterList[0]
                info[pointer.i][pointer.j].level = currentLevel
                if pointer.i+1<nx and maze[pointer.i+1,pointer.j]==1 and info[pointer.i+1][pointer.j].level==-1:
                    info[pointer.i+1][pointer.j].parent = [pointer.i, pointer.j]
                    iterList.append(info[pointer.i+1][pointer.j])
                if pointer.i>0 and maze[pointer.i-1,pointer.j]==1 and info[pointer.i-1][pointer.j].level==-1:
                    info[pointer.i-1][pointer.j].parent = [pointer.i, pointer.j]
                    iterList.append(info[pointer.i-1][pointer.j])
                if pointer.j+1<ny and maze[pointer.i,pointer.j+1]==1 and info[pointer.i][pointer.j+1].level==-1:
                    info[pointer.i][pointer.j+1].parent = [pointer.i, pointer.j]
                    iterList.append(info[pointer.i][pointer.j+1])
                if pointer.j>0 and maze[pointer.i,pointer.j-1]==1 and info[pointer.i][pointer.j-1].level==-1:
                    info[pointer.i][pointer.j-1].parent = [pointer.i, pointer.j]
                    iterList.append(info[pointer.i][pointer.j-1])
        [iterX, iterY] = info[endX-1][endY-1].parent
        path = [np.array([endX,endY])]
        while iterX!=-1 and iterY!=-1:
            path = [np.array([iterX+1,iterY+1])] + path
            [iterX ,iterY] = info[iterX][iterY].parent
        difficulty = 0
        temp_maze = maze
        for i,point in enumerate(path): 
            if i==0 or i==len(path)-1:
                continue
            temp = difficulty
            difficulty+=1
            if point[0]<nx and maze[point[0]][point[1]-1]==0:
                difficulty+=1               
            if point[0]-1>0 and maze[point[0]-2][point[1]-1]==0:
                difficulty+=1              
            if point[1]+1<ny and maze[point[0]-1][point[1]]==0:
                difficulty+=1               
            if point[1]-1>0 and maze[point[0]-1][point[1]-2]==0:
                difficulty+=1    
            if path[i+1][0]!=path[i-1][0] and path[i+1][1]!=path[i-1][1]:
                difficulty+=1
            #print(point, difficulty-temp,maze[point[0]-1][point[1]-1])
        for i,point in enumerate(path): 
            temp_maze[point[0]-1][point[1]-1]=2
        print("难度",difficulty)
        for i,point in enumerate(path):
            path[i] = path[i]*self.grid_wid + self.ref_orig + 1/2*self.grid_wid*np.array([1,1]) 
        self.target_traj = np.array(path)
        #print(self.target_traj)
        #print(temp_maze)

    def calculateTargetBasedOnPredefinedTraj(self):
        n = self.target_traj.shape[0]
        current_block_center = np.floor((self.massCenter - self.ref_orig)/self.grid_wid) * self.grid_wid + self.ref_orig + 1/2*self.grid_wid*np.array([1,1])
        print("current mass at ",self.massCenter, "block center at ",current_block_center)
        d_min = 10
        ind = -1
        for i in range(n):
            d = np.linalg.norm(self.target_traj[i,:]-current_block_center)
            if d<1e-3:  # 当前质心所在的网格是目标路径上的,下一个目标取为该路径的下一个网格中心
                if i+1<n:
                    self.targetPoint = self.target_traj[i+1,:] # 下一个网格中心还有值
                else:
                    self.targetPoint = self.initTarget # 当前所在的网格中心已经是最后一个路径上的点了,取起始输入的目标点
                return
            else:  # 当前质心所在的格点不在路径上
                if d<d_min:
                    ind = i
                    d_min = d
        self.targetPoint = self.target_traj[ind,:]

    """
    def initialize_test(self, target=np.array([0, 0])):
        print("reading maze...")
        self.readMazeAndPlanPath(target=target)
        print("read in maze info and plan path successfully!")
        self.calculateTargetBasedOnPredefinedTraj()
    """

    def initialize(self, target=np.array([0, 0]),autoPilot=False):
        self.initTarget = target #初始目标点
        # 初始物体状态，读入vertex文件中的构成物体的拉格朗日点
        with open(filename + '.vertex', 'r') as f:
            lag = np.loadtxt(f.name, unpack=True, skiprows=1)
        lag = lag.T[:159]
        n=len(lag)
        # 把点的顺序改成从左到右的
        for i in range(41):
            temp=lag[i].copy()
            lag[i] = lag[79-i].copy()
            lag[79-i]=temp.copy()
        mass_center = np.mean(lag,axis=0)

        if autoPilot==False:
            try:
                print("reading maze...")
                self.readMazeAndPlanPath(target=target)
                print("read in maze info and plan path successfully!")
                self.calculateTargetBasedOnPredefinedTraj()
            except:
                # 初始化目标点
                print("normal track")
                self.targetPoint = target
        elif autoPilot==True:
            self.targetPoint = target
            self.SenseSurroundingAreaAndChangeCourse()

        self.pre_targetPoint = self.targetPoint.copy()
        self.targetList.append(target) #记录每个时刻目标点
        pre_state = np.array([lag[0][0], lag[0][1], lag[n//2][0], lag[n//2][1], lag[-1][0], lag[-1][1], self.targetPoint[0], self.targetPoint[1], mass_center[0], mass_center[1], 0, 0, 0, 3])
        curr_state = np.array([lag[0][0], lag[0][1], lag[n//2][0], lag[n//2][1], lag[-1][0], lag[-1][1], self.targetPoint[0], self.targetPoint[1], mass_center[0], mass_center[1], 0, 0, 0, 0])
        realfinal=np.append(pre_state, curr_state)
        return realfinal

    
       


    def plot_traj(self):
        with open('massCenter.txt','w+') as f:
            for element in self.massCenterList:
                 f.write(str(element[0]) + ' ' +str(element[1]) + '\n')
        with open('targetTraj.txt','w+') as f:
            for element in self.targetList:
                 f.write(str(element[0]) + ' ' + str(element[1]) + '\n')
        import matplotlib.pyplot as plt
        from matplotlib import rcParams
        rcParams['font.size'] = 14
        mass = np.array(self.massCenterList)
        target = np.array(self.targetList)
      
        plt.plot(mass[:,0], mass[:,1], label='mass center trajectory',color="blue")

        plt.plot(target[:,0], target[:,1], label='target trajectory',color="green") #临时

        plt.scatter(self.initTarget[0], self.initTarget[1],color="red",s=40,marker='o')
        plt.scatter(self.start_point[0], self.start_point[1],color="red",s=40,marker='o')
        """
        if target[0][0]!=target[3][0] and target[0][1]!=target[3][1]:
            plt.plot(target[:,0], target[:,1], label='target trajectory',color="green")
            n=target.shape[0]
            num=10
            for i in range(2,num+1):
                temp=min(n,n//num*i)
                plt.scatter(target[temp][0], target[temp][1],color="green",s=20,marker='o')
                plt.scatter(mass[temp][0], mass[temp][1],color="blue",s=20,marker='o')
        """
        with open(filename+'.vertex', 'r') as f:
            lag = np.loadtxt(f.name, unpack=True, skiprows=1)
        lag = lag.T
        if lag.shape[0]>159:
            plt.scatter(lag[159:,0], lag[159:,1],color="black",s=4,marker='o')
        if self.knownWallPoints is not None:
            for point in self.knownWallPoints:
                plt.scatter(point[0], point[1],color="yellow",s=4,marker='o')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim([0.1,3.1])
        plt.ylim([0.1,3.1])
        ax=plt.gca()
        ax.set_aspect(1) #bigger for y longer
        plt.legend(loc="best")
        plt.grid(True)
        plt.show()
        plt.savefig("trajectory.png")
        plt.close()

    # 输入action后根据当前的流场信息求解下一步的流场并更新当前的流场状态
    # action 是一个tensor
    # 并返回s_:下一时刻的拉格朗日点的速度+质心3x3的速度分量张量
    # 奖励正比于当前质心与目标点距离的相反数
    def step(self, action):
        with open('../context.txt','r') as f:
            lines=f.readlines()
        with open('../context.txt','w+') as f:
            temp=lines[0].split()
            temp[0]=str(action[0].item())
            temp[1]=str(action[1].item())
            lines[0]=' '.join(temp)+'\n'
            for line in lines:
                f.write(line)

        # 修改input2d中的终止参数
        def update_input2d(line: str) -> Tuple[bool, str]:
            tmp = line.split()
            if len(tmp) > 0:
                if tmp[0] == 'END_TIME':
                    #print('currenTime='+str(self.currentTime)+'正在将tfinal修改为'+str(self.currentTime+self.dt)+'\n')
                    return True, f"END_TIME = {str(self.currentTime+self.dt)}\n"  #去掉了+0.001
            return True, line

        self.update_file_with_line_func('input2d', update_input2d)

        # 脚本调用IBAMR
        command_line = "./main2d input2d"
        if self.currenTimeStep!=0:
            command_line = command_line + " ./restart_IB2d/ "+str(self.currenTimeStep)
        os.system(command_line)


        self.currenTimeStep += step_size
        self.currentTime += self.dt
        self.currstep += 1

        # 记录原先位置
        pre_massCenter = self.massCenter
        pre_lag = self.rearrange_new(self.Lagpoints)
        pre_force = np.sum(self.Force, axis=0)
        pre_torque = self.torque
        
        # 更新环境状态
        self.Lagpoints = self.read_Lagpoint(self.currenTimeStep)  # 读入当前时间步的拉格朗日点
        self.Force = self.read_force(self.currenTimeStep)  # 计算拉格朗日点受力受力
        curr_force = np.sum(self.Force, axis=0)
        self.torque = self.calculate_torque(self.Lagpoints, self.massCenter, self.Force) 
        curr_lag = self.rearrange_new(self.Lagpoints)
        self.massCenter = self.calculate_mass_center(self.Lagpoints)  # 计算质心坐标
        self.massCenterList.append(self.massCenter)
        self.targetList.append(np.array([self.targetPoint[0],self.targetPoint[1]]))
        
        distance=np.linalg.norm(self.targetPoint - self.start_point) #初始位置和目标位置的距离

        n = len(pre_lag)
        self.new_obs = np.array([pre_lag[0][0], pre_lag[0][1], pre_lag[n//2][0], pre_lag[n//2][1], pre_lag[-1][0], pre_lag[-1][1], \
                                 self.pre_targetPoint[0], self.pre_targetPoint[1], pre_massCenter[0], pre_massCenter[1], pre_force[0], pre_force[1], pre_torque, (self.currstep+3)%4, \
                                curr_lag[0][0], curr_lag[0][1], curr_lag[n//2][0], curr_lag[n//2][1], curr_lag[-1][0], curr_lag[-1][1], \
                                    self.targetPoint[0], self.targetPoint[1], self.massCenter[0], self.massCenter[1], curr_force[0], curr_force[1], self.torque, self.currstep%4])

        xxx = curr_lag[:,0]
        yyy = curr_lag[:,1]
        # 判断终止
        if self.currentTime < self.endTime and max(xxx)<3.1 and min(xxx)>0.1 and max(yyy)<3 and min(yyy)>0.1 and np.linalg.norm(self.massCenter - self.targetPoint)> 0.03:
            done = 0
        else:
            done = 1
            # 还要把input2d中的终止时间改回初始值
            def recover_input2d(line: str) -> Tuple[bool, str]:
                tmp = line.split()
                if len(tmp) > 0:
                    if tmp[0] == 'END_TIME':
                        return True, f"END_TIME = {str(self.endTime)}\n"
                return True, line

            self.update_file_with_line_func('input2d', recover_input2d)

        s_ = torch.tensor(self.new_obs, dtype=torch.float64)
        reward = calculate_reward_from_raw_state(s_)
        return s_, reward, done




