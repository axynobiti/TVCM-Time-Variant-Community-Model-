"""
This is an add-on for the TVCM_simple model that makes friend groups meet up sometimes.
Instead of everyone just wandering randomly, we pick groups of friends and have them "meet" in the same spot for a while.

How it works:
1. At the start, we:
   - Keep a copy of the original model so we don't mess it up.
   - Randomly create "friend groups" by picking nodes.
   - Give each group one or more "meeting hours" per day, making sure nobody has two meetings at once.
   - Set up tracking so we know who's currently at a meeting.
2. Every time we update (step):
   - Check if we've moved into a new hour.
   - If yes, for each group that meets now:
     * Find the average location of its members.
     * Instantly move everyone there and freeze them (so they hang out together).
     * Mark them as "in meeting" until the hour ends.
   - Let your original model move everyone (frozen folks stay put).
   - After moving, anyone whose meeting is over gets a new random speed and goes back to wandering.
3. Contact checks are still done by your original model (we just pass through).

"""
import numpy as np
from TVCM_simple import SimpleTVCM

class SocialTVCM:

    def __init__(
        self,
        base: SimpleTVCM,
        p_edge: float = 0.2,
        min_group: int = 2,
        max_group: int = 10,
        slot_len: float = 3600.0,
        n_slots: int = 24,
        meetings_per_day: int = 2,
    ):
        # 1) Save the original model and basics
        self.base = base                           # original TVCM
        self.home_of_node = base.home_of_node      # keep home info
        self.slot_len = slot_len                   # seconds per meeting slot
        self.n_slots = n_slots                     # how many slots in a day
        self.meetings_per_day = meetings_per_day   # times each group meets daily
        self.t = 0.0                               

        # 2) Make friend groups (randomly)
        N = len(self.home_of_node)
        self.groups = []                           # list of sets of node IDs
        for u in range(N):
            # decide group size (random but bounded)
            size = min(min_group + np.random.geometric(p_edge) - 1, max_group)
            # pick random friends
            members = set(np.random.choice(N, size, replace=False))
            members.add(u)                         # don't forget the creator
            self.groups.append(members)

        # 3) Schedule meetups so nobody double-books
        used_colors = {u: set() for u in range(N)}
        self.group_slots = []  # each group gets a list of slot numbers
        spacing = n_slots // meetings_per_day
        for i, members in enumerate(self.groups):
            c = 0
            # find a color (number) none of these members used yet
            while any(c in used_colors[u] for u in members):
                c += 1
            # spread that color into actual hours
            slots = [(c + k * spacing) % n_slots for k in range(meetings_per_day)]
            self.group_slots.append(slots)
            for u in members:
                used_colors[u].add(c)

        # 4) Track who's currently at a meetup (None = wandering)
        self.active = {u: None for u in range(N)}

    def step(self, dt: float = 1.0):
        prev_slot = int(self.t // self.slot_len) % self.n_slots
        self.t += dt
        cur_slot  = int(self.t // self.slot_len) % self.n_slots

        # A) New hour, check for meetups
        if cur_slot != prev_slot:
            for i, slots in enumerate(self.group_slots):
                if cur_slot in slots:
                    members = self.groups[i]
                    # find average position of group members
                    positions = self.base.pos[list(members)]
                    centroid = positions.mean(axis=0)
                    # move them there and freeze
                    for u in members:
                        self.base.pos[u] = centroid.copy()
                        self.base.v[u]   = 0.0
                        self.active[u]   = i  # mark as in meeting

        # B) let the original TVCM-walk run (frozen ones stay put)
        pos = self.base.step(dt)

        # C) check if any meetings ended, and un-freeze folks
        for u, grp in list(self.active.items()):
            if grp is not None:
                if (int(self.t // self.slot_len) % self.n_slots) not in self.group_slots[grp]:
                    self.active[u] = None
                    self.base.v[u] = self.base.rng.uniform(*self.base.v_range)

        return pos

    def contacts(self, positions=None):
        pos = positions if positions is not None else self.base.pos
        return self.base.contacts(pos)
