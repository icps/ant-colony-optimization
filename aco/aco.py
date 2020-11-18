import numpy as np
import pandas as pd
import random as rnd

rnd.seed(321)
np.random.seed(321)

DEBUG = False


class ACO:
    
    def __init__(self, data):
        
        self.edges    = data
        self.vertices = set(data.iloc[:, 1].values.tolist() + data.iloc[:, 2].values.tolist())
    
    
    def calculate_probabilities(self, possible_edges, alpha, beta):
        # (prob inicial ** conc. feromônio (alpha)) * (peso da aresta ** qual. do movimento (beta))
        probs = (possible_edges["ph"] ** alpha) * ((possible_edges["w"]) ** beta)
        return list(probs / sum(probs))
    
    
    def validate_path(self, ant_path, last_node):
        
        if DEBUG:
            print("Validating path...")
        
        ### testar se desse caminho se chega no ultimo vértice
        # vai retirando nó do final do caminho até achar uma aresta do último nó atual do caminho até o nó final
        possible_edges = self.edges[self.edges["n2"].values == last_node]
        
        acceptable_path = False
        finished_path   = False
        while finished_path == False and acceptable_path == False:
                        
            last_node_path = ant_path[-1]
            
            if last_node_path not in list(possible_edges["n1"]):
                ant_path.pop()
                
            else:
                ant_path.append(last_node)
                acceptable_path = True
            
            if len(ant_path) == 0:
                finished_path = True
        
        return ant_path
    
    
    def get_nodes(self, first_node, last_node):
        
        if first_node == "random" and last_node != "random":
            possible_nodes = self.vertices - set([last_node])
            first_node     = rnd.choice(list(possible_nodes))
            
        elif first_node != "random" and last_node == "random":
            possible_nodes = self.vertices - set([first_node])
            last_node      = rnd.choice(list(possible_nodes))
            
        elif first_node == "random" and last_node == "random":
            first_node = rnd.choice(list(self.vertices))
            
            possible_nodes = self.vertices - set([first_node])
            last_node      = rnd.choice(list(possible_nodes))
            
        if DEBUG:
            print(f'FROM {first_node} TO {last_node}')
            
        return first_node, last_node
            
                
    def build_solution(self, first_node, last_node, alpha, beta):
        
        if DEBUG:
            print("Building solution...")
        
        if first_node == "random" or last_node == "random":
            first_node, last_node = self.get_nodes(first_node, last_node)
            
        ant_path = [first_node]
        
        ### a partir do primeiro nó, quais nós são possíveis atingir? 
        possible_edges = self.edges[self.edges["n1"].values == first_node]
                
        ## construir caminho
        finished_path = False
        while finished_path == False:
            
            ant_move  = self.calculate_probabilities(possible_edges, alpha, beta)
            next_node = rnd.choices(list(possible_edges["n2"]), ant_move)[0]          
                       
            ant_path.append(next_node)               
            
            possible_edges = self.edges[self.edges["n1"].values == next_node]
                        
            # exclui nós que já existem das possibilidades de transição, para evitar ciclos
            possible_edges = possible_edges[~possible_edges["n2"].isin(ant_path)]
            
            if possible_edges.empty or next_node == last_node:
                finished_path = True                
        
        if ant_path[-1] != last_node:
            ant_path = self.validate_path(ant_path, last_node)
        
        ant_path = list(zip(ant_path, ant_path[1:]))

        return ant_path
    
    
    def get_cost(self, ant_path):
        ## calcula o custo do caminho
        
        if DEBUG:
            print("Getting cost...")
        
        total_cost = 0
        
        for e1, e2 in ant_path:
            edge = self.edges[self.edges["n1"].values == e1]
            edge = edge[edge["n2"].values == e2]
            
            total_cost = total_cost + edge["w"].values[0]
            
        return total_cost
        
    
    def update_pheromone(self, solutions, evaporation_rate):
        # se a formiga passou numa aresta, ela deixa feromônio
        
        if DEBUG:
            print("Updating pheromone...")
        
        self.edges["ph"] = self.edges["ph"] * (1 - evaporation_rate)

        for current_solution in solutions:
            cost, path = current_solution

            for e1, e2 in path:
                edge          = self.edges[self.edges["n1"].values == e1]
                current_index = edge[edge["n2"].values == e2].index

                self.edges.loc[current_index, "ph"] = self.edges.loc[current_index, "ph"] + (1 / cost)
        
        if DEBUG:
            print("Pheromone updated...")
    
    
    def update_global_pheromone(self, global_best, local_best, current_ants, evaporation_rate, update):
        
        if update == "global":
            self.update_pheromone([global_best], evaporation_rate)
                
        elif update == "local":
            self.update_pheromone([local_best], evaporation_rate)
            
        elif update == "global-local":
            solutions = []
            solutions.append(global_best)
            solutions.append(local_best)

            self.update_pheromone(solutions, evaporation_rate)
            
        elif update == "all":
            solutions = [ant for ant in current_ants]            
            self.update_pheromone(solutions, evaporation_rate)   
    
    
    def longest_path(self, first_node, last_node, pheromone_init, max_iter, 
                     n_ants, alpha, beta, evaporation_rate, update): 
        
        ants = {i: None for i in range(1, max_iter + 1)}
        
        self.edges["ph"] = pheromone_init
        global_best      = (np.NINF, [])
        
        for gen in range(1, max_iter + 1):
            
            local_best   = (np.NINF, [])    # (fitness, solution)
            current_ants = []            
                        
            for ant in range(n_ants):
                ant_path = self.build_solution(first_node, last_node, alpha, beta)
                
                path_cost = self.get_cost(ant_path)
                
                current_ants.append((path_cost, ant_path))
                
                if path_cost > local_best[0]:
                    local_best = (path_cost, ant_path)
                    
                if path_cost > global_best[0]:
                    global_best = (path_cost, ant_path)
                    
                if DEBUG:
                    print("--- One more ant...")
                    
            ants[gen] = pd.DataFrame(current_ants, columns = ['cost', 'path'])
            
            self.update_global_pheromone(global_best, local_best, current_ants, evaporation_rate, update)
            
        print("Best solution: {} -- {}".format(global_best[0], global_best[1]))
        
        return ants