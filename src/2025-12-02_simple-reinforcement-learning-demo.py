# python
import numpy as np
import random

class FAKEnv:
    """
    Single-tail, single-part environment for FAK decisions.
    State: (leg_idx, inventory_remaining)
    Action: 0 = do not carry, 1 = carry
    Reward: negative costs per leg and failure outcomes
    """

    def __init__(self, failure_probs, has_spares, init_inventory,
                 c_carry=5.0, c_repair=50.0, c_aog=1000.0, seed=42):
        """
        failure_probs: list of length N, probability of failure on each leg
        has_spares: list of length N, bool or 0/1 for local spares availability at destination
        init_inventory: initial integer inventory at base
        costs: c_carry (per leg), c_repair (if failure handled), c_aog (if failure cannot be handled)
        """
        assert len(failure_probs) == len(has_spares), "Mismatch in schedule lengths"
        self.N = len(failure_probs)
        self.failure_probs = np.array(failure_probs, dtype=float)
        self.has_spares = np.array(has_spares, dtype=int)
        self.init_inventory = int(init_inventory)
        self.c_carry = float(c_carry)
        self.c_repair = float(c_repair)
        self.c_aog = float(c_aog)
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.leg = 0
        self.inventory = self.init_inventory  # total inventory available anywhere
        # Note: simplified assumption: using part reduces total inventory by 1 (FAK or local)
        return (self.leg, self.inventory)

    def step(self, action):
        """
        Execute action at current leg:
        - action=1 (carry): pay carry cost; if no inventory, action is forced to 0 (can't carry).
        - sample failure; apply costs accordingly; consume inventory if repair happens.
        Returns: next_state, reward, done, info
        """
        # Validate action (can't carry if no inventory)
        can_carry = (self.inventory > 0)
        a = int(action)
        if a == 1 and not can_carry:
            a = 0  # override

        # Costs
        reward = 0.0
        if a == 1:
            reward -= self.c_carry

        # Simulate failure
        fail = (self.rng.random() < self.failure_probs[self.leg])

        if fail:
            if a == 1:
                # FAK covers failure: consume inventory; pay repair
                reward -= self.c_repair
                self.inventory = max(0, self.inventory - 1)
            else:
                if self.has_spares[self.leg] == 1:
                    # Local spares handle failure: consume inventory; pay repair
                    reward -= self.c_repair
                    self.inventory = max(0, self.inventory - 1)
                else:
                    # No FAK and no local spares: AOG
                    reward -= self.c_aog

        # Advance leg
        self.leg += 1
        done = (self.leg >= self.N)
        next_state = (self.leg, self.inventory)
        return next_state, reward, done, {"fail": fail, "carried": (a == 1)}

    def state_space(self):
        # Leg index [0..N], inventory [0..init_inventory]
        return (self.N + 1, self.init_inventory + 1)

    def action_space(self):
        return 2  # 0 or 1


def q_learning(env, episodes=5000, alpha=0.2, gamma=0.95, eps_start=1.0, eps_end=0.05, eps_decay=0.9995, seed=7):
    rng = random.Random(seed)
    S_leg, S_inv = env.state_space()
    A = env.action_space()

    # Q-table: [leg_idx][inventory][action]
    Q = np.zeros((S_leg, S_inv, A), dtype=float)

    eps = eps_start
    rewards_history = []

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            leg, inv = state

            # Epsilon-greedy
            if rng.random() < eps:
                action = rng.randrange(A)
            else:
                action = int(np.argmax(Q[leg, inv, :]))

            next_state, reward, done, info = env.step(action)
            total_reward += reward

            # Q-update
            nleg, ninv = next_state
            best_next = 0.0 if done else np.max(Q[nleg, ninv, :])
            td_target = reward + gamma * best_next
            td_error = td_target - Q[leg, inv, action]
            Q[leg, inv, action] += alpha * td_error

            state = next_state

        # Decay epsilon
        eps = max(eps_end, eps * eps_decay)
        rewards_history.append(total_reward)

    return Q, rewards_history


def derive_policy(Q):
    """
    Convert Q-table to a deterministic policy: for each state, pick action with max Q-value.
    """
    S_leg, S_inv, A = Q.shape
    policy = np.zeros((S_leg, S_inv), dtype=int)
    for leg in range(S_leg):
        for inv in range(S_inv):
            policy[leg, inv] = int(np.argmax(Q[leg, inv, :]))
    return policy


def simulate_policy(env, policy, runs=1000):
    """
    Evaluate policy by simulation with stochastic failures.
    """
    total = 0.0
    for _ in range(runs):
        state = env.reset()
        done = False
        tr = 0.0
        while not done:
            leg, inv = state
            action = policy[leg, inv]
            next_state, reward, done, info = env.step(action)
            tr += reward
            state = next_state
        total += tr
    return total / runs


if __name__ == "__main__":
    # Example data
    # 8 legs; remote legs are at indices where has_spares=0
    failure_probs = [0.02, 0.03, 0.05, 0.20, 0.04, 0.01, 0.15, 0.03]
    has_spares =     [1,    1,    0,    0,    1,    1,    0,    1]
    init_inventory = 2

    # Costs
    c_carry = 5.0      # carrying penalty per leg
    c_repair = 50.0    # repair cost when failure is handled
    c_aog = 1000.0     # very large cost when failure cannot be handled (AOG)

    env = FAKEnv(failure_probs, has_spares, init_inventory, c_carry, c_repair, c_aog, seed=123)

    # Train Q-learning
    Q, rewards = q_learning(env, episodes=8000, alpha=0.25, gamma=0.95,
                            eps_start=1.0, eps_end=0.05, eps_decay=0.999, seed=42)

    policy = derive_policy(Q)
    avg_cost = simulate_policy(env, policy, runs=2000)

    print("Average total cost per schedule (learned policy):", avg_cost)

    # Compare to a simple heuristic: carry only on remote legs if inventory > 0
    def heuristic_sim(env, runs=2000):
        total = 0.0
        for _ in range(runs):
            state = env.reset()
            done = False
            tr = 0.0
            while not done:
                leg, inv = state
                action = 1 if (env.has_spares[leg] == 0 and inv > 0) else 0
                next_state, reward, done, info = env.step(action)
                tr += reward
                state = next_state
            total += tr
        return total / runs

    avg_cost_heur = heuristic_sim(env, runs=2000)
    print("Average total cost per schedule (heuristic policy):", avg_cost_heur)