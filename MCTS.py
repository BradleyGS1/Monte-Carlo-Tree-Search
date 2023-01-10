import random
import math
import numpy as np
import networkx as nx
import pygraphviz as pgv

# Define Noughts and Crosses
class NoughtsCrosses(object):
    def __init__(self):
        # Initialise whose turn it is
        self.turn = True
        # Initialise game board 
        self.board = np.full((3, 3), fill_value = 0)

    def get_legal_moves(self):
        legal_moves = []
        for i in range(0, 3):
            for j in range(0, 3):
                move = (i, j)
                if self.board[i, j] == 0:
                    legal_moves.append(move)

        return legal_moves
    
    def push(self, move:tuple):
        self.board[move[0], move[1]] = 2 * self.turn - 1
        self.turn = not self.turn 

    def winner(self, legal_moves:list = None):
        three_in_row = [i in np.sum(self.board, axis = 0) for i in [-3, 3]]
        three_in_col = [i in np.sum(self.board, axis = 1) for i in [-3, 3]]
        three_in_diag = [i in [np.trace(self.board), np.trace(np.fliplr(self.board))] for i in [-3, 3]]
        
        if legal_moves == None:
            legal_moves = self.get_legal_moves()

        last_player = not self.turn
        if three_in_row[last_player] or three_in_col[last_player] or three_in_diag[last_player]:
            return last_player

        elif len(legal_moves) == 0:
            return "Draw"

        else:
            return "None"

class MCTS(object):
    def __init__(self):
        # Initialise tree
        self.tree = nx.DiGraph()
        self.tree.add_node(0, move = "*", wins = 0, visits = 0, ucb = 0)

    def train(self, env:object, num_samples:int = 100, save_img:bool = False):
        tree = self.tree
        for _ in range(0, num_samples):
            # Re-initialise the environment
            env.__init__()
            legal_moves = env.get_legal_moves()

            # Initialise list of visited nodes
            # Players will alternate ownership of the visited nodes in this list
            visited_nodes = [0]

            # Selection Stage 
            # Move down the tree until we hit a leaf node or game ends
            # A leaf is any node with a potential child which hasn't been simulated yet
            # If all legal moves are already in the tree then we need to keep searching

            child_nodes = tree.successors(0)
            child_moves = [tree.nodes[child_node]["move"] for child_node in child_nodes]
            winner = "None"
            while all([move in child_moves for move in legal_moves]) and winner == "None":
                # Using the Upper Confidence Bound (UCB) to select child nodes
                ucb_values = {}
                for child_node in tree.successors(visited_nodes[-1]):
                    ucb_values[child_node] = tree.nodes[child_node]["ucb"]

                # Get node to visit (choose node with maximum UCB value)
                new_node = max(ucb_values, key = ucb_values.get)
                # Add node to visited nodes list
                visited_nodes.append(new_node)

                # Get new iterator of child nodes from this new node
                child_nodes = tree.successors(visited_nodes[-1])
                # Get list of moves represented by the child nodes from this new node
                child_moves = [tree.nodes[child_node]["move"] for child_node in child_nodes]

                # Update the environment, get new legal moves and check for game over
                new_move = tree.nodes[new_node]["move"]
                env.push(new_move)
                legal_moves = env.get_legal_moves()
                winner = env.winner(legal_moves = legal_moves)

            # Exploration Stage
            # Chooses child node from the current leaf node of the tree at random
            # Unless the current leaf node has already ended the game decisively
            # Add this new child node to the tree if not already on and visited nodes list

            if winner == "None":
                # Pick a random legal_move from the current environment state
                move = random.choice(legal_moves)

                # If the current node has no children which represents this new move
                # then we add a new child to the current node
                if move not in child_moves:
                    new_node = tree.number_of_nodes()
                    # Add the new node and edge with attributes to the current leaf node
                    tree.add_node(new_node, move = move, wins = 0.0, visits = 0.0, ucb = 0.0)
                    tree.add_edge(visited_nodes[-1], new_node)
                
                # Add the new node to the visited nodes list
                visited_nodes.append(new_node)
                # Push this new move to the environment
                env.push(move)
                # Get new legal moves
                legal_moves = env.get_legal_moves()
                # Check if there is now a winner
                winner = env.winner(legal_moves = legal_moves)
            
            # Simulation Stage
            # From this new node, complete moves randomly until the game is decided

            while winner == "None":
                move = random.choice(legal_moves)
                env.push(move)
                legal_moves = env.get_legal_moves()
                winner = env.winner(legal_moves = legal_moves)
            
            # Backpropogation
            # Given the result of the simulation stage we can backpropogate the win
            # rewards back towards the visited nodes respective of the player who visited

            for i in range(0, len(visited_nodes)):
                node = visited_nodes[i]

                tree.nodes[node]["visits"] += 1.0
                if winner == "Draw":
                    tree.nodes[node]["wins"] += 0.5

                elif (winner == True) and (i % 2 == 0):
                    tree.nodes[node]["wins"] += 1.0    

                elif (winner == False) and (i % 2 == 1):
                    tree.nodes[node]["wins"] += 1.0

            # Update the win, visit, ucb values for all visited nodes
            for node in visited_nodes[1:]:
                # Get needed attributes from visited node and its parent
                node_wins = tree.nodes[node]["wins"]
                node_visits = tree.nodes[node]["visits"]
                # !!!Check if the below line is working, as there seems to be a glitch!!!
                parent_node = list(tree.predecessors(node))[0]
                parent_visits = tree.nodes[parent_node]["visits"]

                # Calculate the new ucb value for this visited node
                ucb_value = (
                    node_wins / node_visits 
                    + 
                    math.sqrt(2 * math.log(parent_visits) / node_visits)
                )
                # Update the visited nodes ucb value
                tree.nodes[node]["ucb"] = ucb_value

        if save_img == True:
            # For each node add a 'label' attribute which stores the info we want
            # to display in the image of the tree
            for node in range(0, tree.number_of_nodes()):
                node_move = tree.nodes[node]["move"]
                node_wins = tree.nodes[node]["wins"]
                node_visits = tree.nodes[node]["visits"]
                tree.nodes[node]["label"] = f"{node_move}\n{node_wins}/{node_visits}"

            # Display Tree
            # Convert networkx tree into pygraphviz tree for better visualisation
            tree_viz = nx.nx_agraph.to_agraph(tree)
            tree_viz.draw("Tree.png", prog = "circo")

        # Update the object initialised tree variable
        self.tree = tree
        print("Tree Fitted")

    def optimal_move(self, prev_moves:list):
        tree = self.tree
        node = 0
        child_nodes = tree.successors(0)
        for move in prev_moves:
            # Get the next node using what the previous moves have been
            for child_node in child_nodes:
                if tree.nodes[child_node]["move"] == move:
                    node = child_node
                    break

            # Update child nodes of new node
            child_nodes = tree.successors(node)
        
        # Now we have the node representing the most recent move played we can
        # get the optimal move by choosing the next node with the most visits
        visits_values = {}
        for child_node in child_nodes:
            visits_values[child_node] = tree.nodes[child_node]["visits"]

        # Get new node (choose node with maximum visits value)
        optimal_node = max(visits_values, key = visits_values.get)

        # Get optimal move
        optimal_move = tree.nodes[optimal_node]["move"]

        return optimal_move



# Initialise Noughts and Crosses game
env = NoughtsCrosses()

# Run Monte Carlo Tree Search
mcts = MCTS()
mcts.train(env, num_samples = 500, save_img = True)
prev_moves = [(0, 0), (1, 0), (0, 1), (1, 1)]
mcts.optimal_move(prev_moves)