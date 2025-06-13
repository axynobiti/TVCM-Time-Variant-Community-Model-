import numpy as np
from scipy.spatial import cKDTree

class SimpleTVCM:
    def __init__(
        self,
        Gx=6, Gy=6,            # grid dimensions (cells in x and y)
        cell=100,              # size of each cell in meters
        n_comm=10,             # number of communities (home cells)
        n_nodes=505,           # total nodes
        p_range=(0.6, 0.95),   # range for home-stay probability p
        v_range=(1.0, 3.0),    # walking speed range (m/s)
        radio=30,              # two nodes are considered ‘in contact’ if they come within 30 meters of each other
        seed=None              
    ):
        # initialize
        self.rng = np.random.default_rng(seed)
        self.Gx, self.Gy = Gx, Gy
        self.cell = cell
        self.radio = radio
        self.v_range = v_range

       
        # We have 36 cells in total so pick 10 of them randomly , these are the communities
        cells = self.rng.choice(Gx * Gy, n_comm, replace=False)

        # Splitiing the nodes evenly among the communities
        sizes = np.full(n_comm, n_nodes // n_comm, dtype=int)
        sizes[0] += n_nodes - sizes.sum()  # handle any remaining nodes

        # Create an array where each node is assigned to a home cell by checking the cell index and repeating it
        # according to the sizes of each community
        self.home_of_node = np.repeat(cells, sizes)

        # assign each node's stay probability 'p' for the Markov chain 
        # by doing that give every node its own “personality” some love their home and other are travellers !
        self.p_stay = self.rng.uniform(*p_range, size=n_nodes)

        # randomly places each node somewhere inside its home cell, 
        # so that at time zero every node has a valid starting point
        self.pos = self._init_positions() 

        # will store the (x,y) target that each node  is currently walking towards
        self.wp = np.zeros_like(self.pos)

        # direction vector towards the waypoint to avoid magically teleporting
        # and to normalize the speed so that each node moves at its own speed
        self.dir = np.zeros_like(self.pos)

        # to store the current speed of each node
        self.v = np.zeros(n_nodes)

        # Pre-compute initial waypoint & speed for each node
        for i in range(n_nodes):
            self._pick_next_waypoint(i)

    def _init_positions(self):
        """""
       Initialize node positions:
        - Each node i has a home cell index: cell_idx = home_of_node[i]
        - Convert that flat index into 2D grid coords (gx,gy)
        - Each cell spans in meters: [gx*cell, (gx+1)*cell] × [gy*cell, (gy+1)*cell]
        - Pick a uniform random point (x,y) inside that cell for node i
        Returns:
            pos: an (N×2) array of float positions, where N = number of nodes
        """
        N = len(self.home_of_node)
        pos = np.zeros((N, 2))
        for i, cell_idx in enumerate(self.home_of_node):
            gx, gy = divmod(cell_idx, self.Gy)
            x = self.rng.uniform(gx * self.cell, (gx + 1) * self.cell)
            y = self.rng.uniform(gy * self.cell, (gy + 1) * self.cell)
            pos[i] = (x, y)
        return pos

    def _pick_next_waypoint(self, i):
        """
        2-state Markov decision:
        - With probability p_stay[i], stay in home cell
        - Otherwise, roam to a random cell
        Then pick a uniform random waypoint in that cell and speed.
        """
        # 2-state Markov: stay-home vs roam
        if self.rng.random() < self.p_stay[i]:
            cell_idx = self.home_of_node[i]
        else:
            # roam — pick among *other* cells only
            home = self.home_of_node[i]
             # create an array or list of possible cells, excluding `home`
            others = np.delete(np.arange(self.Gx * self.Gy), home)
            cell_idx = self.rng.choice(others)

        gx, gy = divmod(cell_idx, self.Gy)
        tx = self.rng.uniform(gx * self.cell, (gx + 1) * self.cell)
        ty = self.rng.uniform(gy * self.cell, (gy + 1) * self.cell)
        self.wp[i] = np.array([tx, ty])

        vec = self.wp[i] - self.pos[i]
        dist = np.linalg.norm(vec)
        if dist < 1e-6:
            return self._pick_next_waypoint(i)
        self.dir[i] = vec / dist
        self.v[i] = self.rng.uniform(*self.v_range)

    def step(self, dt=1.0):
        """
        Advance by dt seconds:
        - Check arrivals, assign new waypoint via Markov
        - Move towards waypoint, clamp overshoot
        Returns updated positions.
        """
        arrived = np.linalg.norm(self.wp - self.pos, axis=1) < 1e-2
        for i in np.where(arrived)[0]:
            self._pick_next_waypoint(i)

        # move nodes
        self.pos += self.dir * self.v[:, None] * dt
        # clamp overshoot
        overshoot = ((self.pos - self.wp) * self.dir).sum(axis=1) > 0
        self.pos[overshoot] = self.wp[overshoot]
        return self.pos.copy()

    def contacts(self, positions):
        """
        Fast detection of node pairs within 'radio' meters.
        Returns set of (i,j) with i<j.
        """
        return cKDTree(positions).query_pairs(self.radio)

