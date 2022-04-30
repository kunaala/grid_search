import numpy as np
import gym
from utils import *

MF = 0 # Move Forward
TL = 1 # Turn Left
TR = 2 # Turn Right
PK = 3 # Pickup Key
UD = 4 # Unlock Door

class Grid_search:
    def __init__(self,env_path,disp = False,rndm=0):
        if rndm==0:
            self.env,self.info = load_env(env_path)
            print('<Environment Info>\n')
            print(self.info) # Map size
                        # agent initial position & direction, 
                        # key position, door position, goal position
            print('<================>\n') 
            self.key_hold = self.env.carrying is not None
            self.door = self.env.grid.get(self.info['door_pos'][0], self.info['door_pos'][1])
            self.door_open = self.door.is_open
            self.key_pos = self.info['key_pos']
            self.goal_pos = self.info['goal_pos']
            self.door_pos = self.info['door_pos']
        else:
            # FOR RANDOM MAPS #
            self.env, self.info, env_path = load_random_env(env_path)
            print('<Environment Info>\n')
            print(env_path,'\n',self.info) # Map size
                        # agent initial position & direction, 
                        # key position, door position, goal position
            print('<================>\n') 
            
            self.key_hold = self.env.carrying is not None
            self.door1 = self.env.grid.get(self.info['door_pos'][0][0], self.info['door_pos'][0][1])
            self.door2 = self.env.grid.get(self.info['door_pos'][1][0], self.info['door_pos'][1][1])

            self.d1_pos = self.info['door_pos'][0]
            self.d2_pos = self.info['door_pos'][1]
            self.d1_open = self.info['door_open'][0]
            self.d2_open = self.info['door_open'][1]
            # key_dict = {0:(1, 1), 1:(2, 3), 2:(1, 6)} 
            # goal_dict ={0:(5, 1), 1:(6, 3), 2:(5, 6)}
            self.key_arr = np.array([[1, 1], [2, 3], [1, 6]])
            self.goal_arr= np.array([[5, 1], [6, 3], [5, 6]])
            for k in range(self.key_arr.shape[0]) : 
                if np.array_equal(self.key_arr[k],self.info['key_pos']) : 
                    self.key_idx = k
                    self.key_pos = self.key_arr[k]
            for k in range(self.goal_arr.shape[0]) : 
                if np.array_equal(self.goal_arr[k],self.info['goal_pos']) : 
                    self.goal_idx =k
                    self.goal_pos = self.goal_arr[k]


            plot_env(self.env)

    def front_pos(self,curr_pos,heading): 
        if heading == 3: # North
            next_pos = np.array([curr_pos[0],curr_pos[1]-1])
        elif heading == 0: # East
            next_pos = np.array([curr_pos[0]+1,curr_pos[1]])
        elif heading == 1: # South
            next_pos = np.array([curr_pos[0],curr_pos[1]+1])
        elif heading == 2: # West
            next_pos = np.array([curr_pos[0]-1,curr_pos[1]])
        return next_pos

    def motion_model(self, curr_pos,heading,door_open, key_hold, action):
        f_pos = self.front_pos(curr_pos,heading)
        
        if action == 0: # Move Forward
            if type(self.env.grid.get(f_pos[0],f_pos[1])).__name__ == "Wall":
            
               return curr_pos,heading,door_open, key_hold 

            elif type(self.env.grid.get(f_pos[0],f_pos[1])).__name__ == "Door" and door_open != 1:
                return curr_pos,heading,door_open, key_hold
            elif np.array_equal(self.info["key_pos"],f_pos) and key_hold == 0:
                return curr_pos,heading,door_open, key_hold
            else:
                return f_pos,heading,door_open, key_hold
        
        elif action == 1 : # Turn Left
            heading  = heading - 1 if heading != 0 else heading + 3
            return curr_pos,heading,door_open, key_hold
        elif action == 2: # Turn Right
            heading  = heading + 1 if heading != 3 else heading - 3
            return curr_pos,heading,door_open, key_hold

        elif action == 3: # Pickup Key
            if np.array_equal(self.info["key_pos"],f_pos):  key_hold = 1
            return curr_pos,heading,door_open, key_hold

        elif action == 4 : # Unlock Door
            if np.array_equal(self.info["door_pos"],f_pos) and key_hold ==1 : door_open= 1
            return curr_pos,heading,door_open, key_hold
        


        
        
    def cost(self,curr_pos, next_pos, heading, key_hold,action):
        
        f_pos = self.front_pos(curr_pos,heading)
        front_type = type(self.env.grid.get(f_pos[0],f_pos[1])).__name__

         
        
        if action == 0 and np.array_equal(next_pos,curr_pos) : 
            return np.inf

        if action == 3 and front_type != "Key" : 
            # print(f_pos)
            return np.inf
        if action == 4 and front_type != "Door" : 

            return np.inf
        
        if action == 4 and front_type == "Door" and key_hold == 0:
            return np.inf

        else :
            return 10
        
        

    def door_key(self):
        h = self.info["height"]
        w = self.info["width"]
        Q = np.full((w,h,4,2,2,5),np.inf)
        V = np.full((w,h,4,2,2),np.inf)
        # maximum timesteps = w*h of the grid ( includes wall cells and blocks to compensate for key picking and door opening)
        P = np.full((w,h,4,2,2,w*h),np.inf) 
        V[self.goal_pos[0], self.goal_pos[1],...] = 0
        # P = V.copy()
        term=False
        # count =0
        for t in range(w*h-1 ,-1,-1):
            print("timestep:<=================================> ",t)
            if term == False:
                # count =count+1
                for i in range(1,Q.shape[0]-1):
                    for j in range(1,Q.shape[1]-1):
                        if i == self.goal_pos[0] and j == self.goal_pos[1]: continue
                        elif type(self.env.grid.get(i,j)).__name__ == "Wall": continue
                        for k in range(Q.shape[2]):
                            for l in range(Q.shape[3]):
                                for m in range(Q.shape[4]):
                                    for n in range(Q.shape[5]):
                                        next_pos,next_heading,door_open,key_hold = self.motion_model(np.array([i,j]),k,l, m,n)
                                        Q[i,j,k,l,m,n] = self.cost(np.array([i,j]),next_pos,k,m,n) + V[next_pos[0],next_pos[1],next_heading,door_open,key_hold]
                                    V[i,j,k,l,m] = min(Q[i,j,k,l,m,:])
                                    P[i,j,k,l,m,t] = np.argmin(Q[i,j,k,l,m,:])
                                    # if t<2 : 
                                    # print("time",t,"value : policy at",i,j,k,l,m,':\t',V[i,j,k,l,m],":",P[i,j,k,l,m,t])
                # if V[self.env.agent_pos[0],self.env.agent_pos[1],self.env.agent_dir,int(self.door_open),int(self.key_hold)] != np.inf: term =True             
            # print(count)
        
        done = False
        opt_cost= V[self.env.agent_pos[0],self.env.agent_pos[1],self.env.agent_dir,int(self.door_open),int(self.key_hold)]
        opt_pol =[]
        plot_env(self.env)
        while not done:
            # print("time:",t,self.env.agent_pos,self.env.agent_dir, self.door_open, self.key_hold)
            # pos,heading, door_open, key_hold = self.motion_model(pos,heading,door_open,key_hold,int(P[self.env.agent_pos[0],self.env.agent_pos[1],self.env.agent_dir,int(self.door_open),int(self.key_hold),t]))
            t =t+1
            opt_pol.append(int(P[self.env.agent_pos[0],self.env.agent_pos[1],self.env.agent_dir,int(self.door_open),int(self.key_hold),t]))
            _, done = step(self.env, int(P[self.env.agent_pos[0],self.env.agent_pos[1],self.env.agent_dir,int(self.door_open),int(self.key_hold),t]))
            
            plot_env(self.env)
            self.key_hold = self.env.carrying is not None
            self.door_open = self.door.is_open
        return opt_pol,opt_cost

    # FOR RANDOM MAPS    

    def rnd_motion_model(self, curr_pos,heading,d1_open, d2_open, key_hold, key_idx,action):
        f_pos = self.front_pos(curr_pos,heading)
        
        if action == 0: # Move Forward
            if type(self.env.grid.get(f_pos[0],f_pos[1])).__name__ == "Wall":
            
               return curr_pos,heading,d1_open, d2_open, key_hold 

            elif np.array_equal(self.d1_pos,f_pos) and d1_open != 1:
                return curr_pos,heading,d1_open, d2_open, key_hold
            elif np.array_equal(self.d2_pos,f_pos) and d2_open != 1:
                return curr_pos,heading,d1_open, d2_open, key_hold
            elif np.array_equal(self.key_arr[key_idx],f_pos) and key_hold == 0:
                return curr_pos,heading,d1_open, d2_open, key_hold
            else:
                return f_pos,heading,d1_open, d2_open, key_hold
        
        elif action == 1 : # Turn Left
            heading  = heading - 1 if heading != 0 else heading + 3
            return curr_pos,heading,d1_open, d2_open, key_hold
        elif action == 2:  # Turn Right
            heading  = heading + 1 if heading != 3 else heading - 3
            return curr_pos,heading,d1_open, d2_open, key_hold

        elif action == 3: # Pickup Key
            if np.array_equal(self.key_arr[key_idx],f_pos):  
                key_hold = 1
            return curr_pos,heading,d1_open, d2_open, key_hold

        elif action == 4 : # Unlock Door
            if np.array_equal(self.d1_pos,f_pos) and key_hold ==1 : d1_open= 1
            if np.array_equal(self.d2_pos,f_pos) and key_hold ==1 : d2_open= 1
            return curr_pos,heading,d1_open, d2_open, key_hold
        
    def rnd_cost(self,curr_pos, next_pos, heading, key_hold,action,goal_idx):
        
        f_pos = self.front_pos(curr_pos,heading)
        front_type = type(self.env.grid.get(f_pos[0],f_pos[1])).__name__

        if np.array_equal(curr_pos,self.goal_arr[goal_idx]) : 
            return 0
        
        if action == 0 and np.array_equal(next_pos,curr_pos) : 
            return np.inf

        if action == 3 and front_type != "Key" : 
            return np.inf
        if action == 4 and front_type != "Door" : 

            return np.inf
        
        if action == 4 and front_type == "Door" and key_hold == 0:
            return np.inf

        else :
            return 10

    # FOR RANDOM MAPS 
    def rnd_door_key(self):
        h = self.info["height"]
        w = self.info["width"]
        Q = np.full((w,h,4,2,2,2,3,3,5),np.inf)
        V = np.full((w,h,4,2,2,2,3,3),np.inf)
        # maximum timesteps = w*h of the grid ( includes wall cells and blocks to compensate for key picking and door opening)
        P = np.full((w,h,4,2,2,2,3,3,w*h),np.inf) 
        for g in range(self.goal_arr.shape[0]): V[self.goal_arr[g,0],self.goal_arr[g,1],...,g]=0
        term=False
        # count =0
        for t in range(w*h-1 ,-1,-1):
            print("timestep:<=================================> ",t)
            if term == False:
                # count =count+1
                for i in range(0,Q.shape[0]-1): # Grid x
                    for j in range(0,Q.shape[1]-1): # Grid y
                        if type(self.env.grid.get(i,j)).__name__ == "Wall": continue
                        for k in range(Q.shape[2]):  # heading
                            for l in range(Q.shape[3]): # door 1
                                for m in range(Q.shape[4]): # door 2
                                    for n in range(Q.shape[5]): # key_hold
                                        for o in range(Q.shape[6]): # key_pos
                                            for p in range(Q.shape[7]): # goal pos
                                                for q in range(Q.shape[8]): # actions
                                                    next_pos,next_heading,d1_open,d2_open, key_hold = self.rnd_motion_model(np.array([i,j]),k,l,m,n,o,q)
                                                    Q[i,j,k,l,m,n,o,p,q] = self.rnd_cost(np.array([i,j]),next_pos,k,n,q,p) + V[next_pos[0],next_pos[1],next_heading,d1_open,d2_open,key_hold,o,p]
                                            # if t== w*h-1 and 
                                            # if l == int(self.d1_open) and m == int(self.d2_open)  and n == int(self.key_hold): print( i,j,k,l,m,n,o,self.goal_pos,"\t",Q[i,j,k,l,m,n,o])
                                        
                                                V[i,j,k,l,m,n,o,p] = min(Q[i,j,k,l,m,n,o,p,:])
                                                P[i,j,k,l,m,n,o,p,t] = np.argmin(Q[i,j,k,l,m,n,o,p,:])
                        # if t ==0 and i==self.env.agent_pos[0] and j== self.env.agent_pos[1]: print("time",t,"value : policy at",i,j,k,l,m,n,o,p,':\t',V[i,j,k,l,m,n,o,p],":",P[i,j,k,l,m,n,o,p,t])
                # if V[self.env.agent_pos[0],self.env.agent_pos[1],self.env.agent_dir,int(self.door_open),int(self.key_hold)] != np.inf: term =True             
            # print(count)
        np.save("opt_pol.npy",P)
        np.save("opt_cost.npy",V)
        # done = False
        # opt_cost= V[self.env.agent_pos[0],self.env.agent_pos[1],self.env.agent_dir,int(self.d1_open),int(self.d2_open),int(self.key_hold),self.key_idx,self.goal_idx]
        # opt_pol =[]
        # while not done:
        #     print("time:",t,self.env.agent_pos[0],self.env.agent_pos[1],self.env.agent_dir,int(self.d1_open),int(self.d2_open),int(self.key_hold))
        #     # pos,heading, door_open, key_hold = self.motion_model(pos,heading,door_open,key_hold,int(P[self.env.agent_pos[0],self.env.agent_pos[1],self.env.agent_dir,int(self.door_open),int(self.key_hold),t]))
        #     t =t+1
        #     opt_pol.append(int(P[self.env.agent_pos[0],self.env.agent_pos[1],self.env.agent_dir,int(self.d1_open),int(self.d2_open),int(self.key_hold),self.key_idx,self.goal_idx,t]))
        #     _, done = step(self.env, int(P[self.env.agent_pos[0],self.env.agent_pos[1],self.env.agent_dir,int(self.d1_open),int(self.d2_open),int(self.key_hold),self.key_idx,self.goal_idx,t]))
            
        #     plot_env(self.env)
        #     print(opt_pol)
        #     self.key_hold = self.env.carrying is not None
        #     self.d1_open = self.door1.is_open
        #     self.d2_open = self.door2.is_open

        # return opt_pol,opt_cost

    def load_pol(self,pol_file = "opt_pol.npy" , cost_file = "opt_cost.npy"):
        P = np.load(pol_file)
        V = np.load(cost_file)
        done = False
        opt_cost= V[self.env.agent_pos[0],self.env.agent_pos[1],self.env.agent_dir,int(self.d1_open),int(self.d2_open),int(self.key_hold),self.key_idx,self.goal_idx]
        print(opt_cost)
        opt_pol =[]
        t= 0
        while not done:
            print("time:",t,self.env.agent_pos[0],self.env.agent_pos[1],self.env.agent_dir,int(self.d1_open),int(self.d2_open),int(self.key_hold))
            # pos,heading, door_open, key_hold = self.motion_model(pos,heading,door_open,key_hold,int(P[self.env.agent_pos[0],self.env.agent_pos[1],self.env.agent_dir,int(self.door_open),int(self.key_hold),t]))
            t =t+1
            opt_pol.append(int(P[self.env.agent_pos[0],self.env.agent_pos[1],self.env.agent_dir,int(self.d1_open),int(self.d2_open),int(self.key_hold),self.key_idx,self.goal_idx,t]))
            _, done = step(self.env, int(P[self.env.agent_pos[0],self.env.agent_pos[1],self.env.agent_dir,int(self.d1_open),int(self.d2_open),int(self.key_hold),self.key_idx,self.goal_idx,t]))
            
            plot_env(self.env)
            print(opt_pol)
            self.key_hold = self.env.carrying is not None
            self.d1_open = self.door1.is_open
            self.d2_open = self.door2.is_open

        return opt_pol,opt_cost


            
def partA():
    '''
        doorkey-5x5-normal.env
        doorkey-6x6-normal.env
        doorkey-8x8-normal.env
        
        doorkey-6x6-direct.env
        doorkey-8x8-direct.env
        
        doorkey-6x6-shortcut.env
        doorkey-8x8-shortcut.env
    '''
    env_path = './envs/doorkey-8x8-shortcut.env'
    g = Grid_search(env_path) # load an environment
    seq, cost = g.door_key() # find the optimal action sequence
    print(seq,cost)
    draw_gif_from_seq(seq, load_env(env_path)[0]) # draw a GIF & save

def partB():
    env_folder = './envs/random_envs'
    g = Grid_search(env_folder,rndm=1)
    
    # seq, cost = 
    g.rnd_door_key()  
    # pol_file = "opt_pol.npy"
    # cost_file = "opt_cost.npy"  
    seq,cost = g.load_pol()

if __name__ == "__main__":
    # please provide the known map in env_path = './envs/doorkey-8x8-shortcut.env' in partA
    # partA()
    partB()
   

     
    


