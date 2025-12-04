# python
import numpy as np
import random
from collections import defaultdict


class FAKEnv:
    """
    Single-tail, single-part environment for FAK decisions.
    State: (leg_idx, inventory_remaining)
    Action: 0 = do not carry, 1 = carry
    Reward: negative costs per leg and failure outcomes
    """

    def __init__(
        self,
        failure_probs,
        has_spares,
        init_inventory,
        c_carry=5.0,
        c_repair=50.0,
        c_aog=1000.0,
        seed=42,
    ):
        """
        failure_probs: list of length N, probability of failure on each leg
        has_spares: list of length N, bool or 0/1 for local spares availability at destination
        init_inventory: initial integer inventory
        costs: c_carry, c_repair, c_aog
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
        self.inventory = self.init_inventory
        return (self.leg, self.inventory)

    def step(self, action):
        """
        Execute action at current leg:
        - action=1 (carry): pay carry cost; if no inventory, action is forced to 0.
        - sample failure; apply costs accordingly; consume inventory if repair happens.
        Returns: next_state, reward, done, info
        """
        can_carry = self.inventory > 0
        a = int(action)
        if a == 1 and not can_carry:
            a = 0  # override

        reward = 0.0
        carried = a == 1
        if carried:
            reward -= self.c_carry

        # Simulate failure
        fail = self.rng.random() < self.failure_probs[self.leg]
        aog = False
        repaired = False
        consumed_inventory = 0

        if fail:
            if carried:
                # FAK covers failure: consume inventory; pay repair
                reward -= self.c_repair
                self.inventory = max(0, self.inventory - 1)
                consumed_inventory = 1
                repaired = True
            else:
                if self.has_spares[self.leg] == 1 and self.inventory > 0:
                    # Local spares handle failure: consume inventory; pay repair
                    reward -= self.c_repair
                    self.inventory = max(0, self.inventory - 1)
                    consumed_inventory = 1
                    repaired = True
                elif self.has_spares[self.leg] == 1 and self.inventory == 0:
                    # Spares exist locally but global pool empty -> AOG (simplified assumption)
                    reward -= self.c_aog
                    aog = True
                else:
                    # No FAK and no local spares: AOG
                    reward -= self.c_aog
                    aog = True

        # Advance leg
        self.leg += 1
        done = self.leg >= self.N
        next_state = (self.leg, self.inventory)
        info = {
            "fail": fail,
            "carried": carried,
            "aog": aog,
            "repaired": repaired,
            "consumed_inventory": consumed_inventory,
        }
        return next_state, reward, done, info

    def state_space(self):
        return (self.N + 1, self.init_inventory + 1)

    def action_space(self):
        # todo: finish this
        return 2  # 0 or 1


def q_learning(
    env,
    episodes=5000,
    alpha=0.2,
    gamma=0.95,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay=0.9995,
    seed=7,
):
    """
    Tabular Q-learning for the FAKEnv.
    Returns Q-table and rewards history across episodes.
    """
    rng = random.Random(seed)
    S_leg, S_inv = env.state_space()
    A = env.action_space()

    Q_state_action_table = np.zeros((S_leg, S_inv, A), dtype=float)

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
                action = int(np.argmax(Q_state_action_table[leg, inv, :]))

            next_state, reward, done, info = env.step(action)
            total_reward += reward

            # Q-update
            # This is equation 19.15 in the Simon Prince book, p384
            nleg, ninv = next_state
            best_next = 0.0 if done else np.max(Q_state_action_table[nleg, ninv, :])
            td_target = reward + gamma * best_next
            td_error = td_target - Q_state_action_table[leg, inv, action]
            Q_state_action_table[leg, inv, action] += alpha * td_error

            state = next_state

        # Decay epsilon
        eps = max(eps_end, eps * eps_decay)
        rewards_history.append(total_reward)

    return Q_state_action_table, rewards_history


def derive_policy_from_Q(Q_state_action_table):
    """
    Convert Q-table to a deterministic policy function:
    policy(state, env) -> action in {0,1}
    """
    S_leg, S_inv, A = Q_state_action_table.shape

    def policy_fn(state, env=None):
        leg, inv = state
        # Clamp to table bounds (useful if someone passes terminal state)
        leg = min(max(leg, 0), S_leg - 1)
        inv = min(max(inv, 0), S_inv - 1)
        return int(np.argmax(Q_state_action_table[leg, inv, :]))

    return policy_fn


def random_policy(state, env):
    """
    Baseline random policy: choose uniformly random feasible action.
    If inventory is 0, only action 0 is feasible.
    """
    leg, inv = state
    if inv <= 0:
        return 0
    return env.rng.randrange(2)  # 0 or 1


def evaluate_policy(env, policy_fn, runs=2000, seed=1234):
    """
    Evaluate a policy on the environment with stochastic failures.
    Returns a dictionary of summary metrics.
    """
    rng = random.Random(seed)
    # We'll re-seed the env per run to vary trajectories.
    totals = []
    metrics = defaultdict(int)

    for r in range(runs):
        # Re-seed env's RNG to get diverse episodes but reproducible across policies
        env.rng = random.Random(rng.randint(0, 10**9))
        state = env.reset()
        done = False
        tr = 0.0
        legs = 0

        while not done:
            action = policy_fn(state, env)
            next_state, reward, done, info = env.step(action)
            tr += reward
            legs += 1

            # Aggregate metrics per leg
            metrics["legs"] += 1
            if info["carried"]:
                metrics["carried_legs"] += 1
            if info["fail"]:
                metrics["fail_legs"] += 1
            if info["aog"]:
                metrics["aog_legs"] += 1
            if info["repaired"]:
                metrics["repaired_legs"] += 1
            metrics["inventory_used"] += info["consumed_inventory"]

            state = next_state

        totals.append(tr)

    # Compute summary stats
    totals_np = np.array(totals, dtype=float)
    out = {
        "avg_total_cost": float(np.mean(totals_np)),
        "std_total_cost": float(np.std(totals_np)),
        "carry_rate": metrics["carried_legs"] / max(1, metrics["legs"]),
        "failure_rate": metrics["fail_legs"] / max(1, metrics["legs"]),
        "aog_rate": metrics["aog_legs"] / max(1, metrics["legs"]),
        "repair_rate": metrics["repaired_legs"] / max(1, metrics["legs"]),
        "avg_inventory_used_per_run": metrics["inventory_used"] / runs,
    }
    return out


def print_metrics(name, metrics):
    """
    Pretty-print evaluation metrics.
    """
    print(f"=== {name} ===")
    print(
        f"Avg total cost: {metrics['avg_total_cost']:.2f}  (std: {metrics['std_total_cost']:.2f})"
    )
    print(f"Carry rate:     {metrics['carry_rate']:.3f}")
    print(f"Failure rate:   {metrics['failure_rate']:.3f}")
    print(f"AOG rate:       {metrics['aog_rate']:.3f}")
    print(f"Repair rate:    {metrics['repair_rate']:.3f}")
    print(f"Avg inventory used/run: {metrics['avg_inventory_used_per_run']:.3f}")
    print()


if __name__ == "__main__":
    # Example data
    failure_probs = [0.02, 0.03, 0.05, 0.20, 0.04, 0.01, 0.15, 0.03]
    has_spares = [1, 1, 0, 0, 1, 1, 0, 1]
    init_inventory = 2

    # Costs
    c_carry = 5.0
    c_repair = 50.0
    c_aog = 1000.0

    # Build environment
    env = FAKEnv(
        failure_probs, has_spares, init_inventory, c_carry, c_repair, c_aog, seed=123
    )

    # Train Q-learning
    Q_trained, rewards = q_learning(
        env,
        episodes=8000,
        alpha=0.25,
        gamma=0.95,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=0.999,
        seed=42,
    )

    # Derive deterministic policy from Q_trained
    learned_policy = derive_policy_from_Q(Q_trained)

    # Evaluate learned policy
    learned_metrics = evaluate_policy(env, learned_policy, runs=4000, seed=999)
    print_metrics("Learned Q-policy", learned_metrics)

    # Evaluate random baseline
    random_metrics = evaluate_policy(env, random_policy, runs=4000, seed=999)
    print_metrics("Random baseline", random_metrics)
