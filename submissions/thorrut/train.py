"""PPO Implementation with CleanRL.

This code is exceedingly basic, with no logging or weights saving.
The intention was for users to have a (relatively clean) ~200 line file to
refer to when they want to design their own learning algorithm.
TODO: custom it

Inspired from this tutorial:
    https://pettingzoo.farama.org/tutorials/cleanrl/implementing_PPO/
    Author: Jet (https://github.com/jjshoots)
"""

from typing import cast
import numpy as np
import torch
import torch.optim as optim
from pettingzoo.mpe import simple_tag_v3
from pettingzoo.utils.env import ParallelEnv
from agent import AdversaryTeamAgent
from prey_agent import StudentAgent as PreyAgent


def batchify_obs(obs: dict, device, agent_order: list[str]) -> torch.Tensor:
    """Converts SimpleTag observations to batch of torch arrays.

    :return obs_b: shape (L, B, observation_feat_nb), with B equaling 1.
    """
    # convert to list of np arrays
    # keep only the predators and remove the two hidden observations
    obs_b = np.stack([obs[agent_id][:-2] for agent_id in agent_order], axis=0)
    # convert to torch
    obs_b = torch.tensor(obs_b).unsqueeze(1).to(device)

    return obs_b


type SimpleTagState = dict[str, np.ndarray] | dict[str, float] | dict[str, bool]


def batchify(x: SimpleTagState, device, agent_order: list[str]) -> torch.Tensor:
    """Converts SimpleTag style returns to batch of torch arrays.

    Let L be the number of agents.

    Arguments:
      dict with for each agent id a ndarray of shape (*,)

    Return:
      shape (L, 1, *), 1 is the batch axis
    """

    # convert to list of np arrays
    b = np.stack([x[agent_id] for agent_id in agent_order], axis=0)
    # convert to torch
    b = torch.tensor(b).unsqueeze(1).to(device)

    return b


def unbatchify(x: torch.Tensor, agent_order: list[str]) -> SimpleTagState:
    """Converts np array to SimpleTag style arguments.

    Let L be the number of agents.

    Arguments:
      tensor of shape (L, 1, *)

    Return:
      a dict with, for each agent id, a ndarray of shape (*,)

    """
    array = x.cpu().numpy()
    return { agent_id: array[k, 0] for k, agent_id in enumerate(agent_order) }


if __name__ == "__main__":
    """ALGO PARAMS"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ent_coef = 0.1
    vf_coef = 0.1
    clip_coef = 0.1
    gamma = 0.99
    batch_size = 32
    # TODO: lengthen the episodes (e.g. to 125)
    max_cycles = 25  # Short episodes for testing
    total_episodes = 2

    """ ENV SETUP """
    num_agents = 3  # 3 adversaries
    env = cast(ParallelEnv, simple_tag_v3.parallel_env(
        num_good=1,  # Number of prey
        num_adversaries=num_agents,  # Number of predators
        num_obstacles=2,
        max_cycles=max_cycles,
        continuous_actions=False
    ))
    def get_agents_from_rest_env(env: ParallelEnv):
        # fix an order for looping over the adversaries
        agent_ids: list[str] = [
            agent_id for agent_id in env.agents if "adversary" in agent_id
        ]
        prey_agent = PreyAgent()
        prey_id: str = [
            agent_id for agent_id in env.agents
            if "adversary" not in agent_id
        ][0]
        return agent_ids, prey_agent, prey_id
    num_actions_per_adversary = 5  # cf. documentation of simple_tag_v3
    observation_size = 14  # cf. documentation of simple_tag_v3

    """ LEARNER SETUP """
    agent = AdversaryTeamAgent(observation_size, num_actions_per_adversary).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=0.001, eps=1e-5)

    """ ALGO LOGIC: EPISODE STORAGE"""
    end_step = 0
    total_episodic_return = 0
    rb_obs = torch.zeros((max_cycles, num_agents, observation_size)).to(device)
    rb_actions = torch.zeros((max_cycles, num_agents)).to(device)
    # we reduce the logprobs, rewards and values for the whole team 
    rb_logprobs = torch.zeros((max_cycles,)).to(device)
    rb_rewards = torch.zeros((max_cycles,)).to(device)
    # we keep this by convention, but for SimpleTag this is not meaningful
    rb_terms = torch.zeros((max_cycles,)).to(device)
    rb_values = torch.zeros((max_cycles,)).to(device)

    """ TRAINING LOGIC """
    # train for n number of episodes
    for episode in range(total_episodes):
        # collect an episode
        with torch.no_grad():
            # collect observations and convert to batch of torch tensors
            next_obs, info = env.reset(seed=None)
            agent_ids, prey_agent, prey_id = get_agents_from_rest_env(env)
            # reset the episodic return
            total_episodic_return = 0

            # each episode has num_steps
            step = 0
            while env.agents:
                # rollover the observation
                obs = batchify_obs(next_obs, device, agent_ids)

                # get action from the agent
                actions, logprobs, _, values = agent.get_on_the_fly_action_and_value_for_team(obs, agent_ids)

                # let the prey act
                prey_obs = next_obs[prey_id]
                prey_action = prey_agent.get_action(prey_obs, prey_id)

                # render the agents' actions
                env_actions = { prey_id: prey_action, **unbatchify(actions, agent_ids) }

                # execute the environment and log data
                next_obs, rewards, terms, truncs, infos = env.step(
                    env_actions
                )

                # add to episode storage
                rb_obs[step] = obs[:, 0]
                rb_rewards[step] = batchify(rewards, device,
                                            agent_ids).sum(dim=0)[0]
                rb_terms[step] = batchify(terms, device,
                                          agent_ids).max(dim=0)[0]
                rb_actions[step] = actions[:, 0]
                rb_logprobs[step] = logprobs[0]
                rb_values[step] = values[0]

                # compute episodic return
                total_episodic_return += rb_rewards[step].cpu().numpy()

                # if we reach termination or truncation, end
                if any([terms[a] for a in terms]) or any([truncs[a] for a in truncs]):
                    end_step = step
                    break
                step += 1
            env.close()

        # bootstrap value if not done
        with torch.no_grad():
            rb_advantages = torch.zeros_like(rb_rewards).to(device)
            for t in reversed(range(end_step)):
                delta = (
                    rb_rewards[t]
                    + gamma * rb_values[t + 1] * rb_terms[t + 1]
                    - rb_values[t]
                )
                rb_advantages[t] = delta + gamma * gamma * rb_advantages[t + 1]
            rb_returns = rb_advantages + rb_values

        # convert our episodes to batch of individual transitions
        b_obs = rb_obs[:end_step]
        b_logprobs = rb_logprobs[:end_step]
        b_actions = rb_actions[:end_step]
        b_returns = rb_returns[:end_step]
        b_values = rb_values[:end_step]
        b_advantages = rb_advantages[:end_step]

        # Optimizing the policy and value network
        b_index = np.arange(b_obs.shape[0])
        clip_fracs = []

        # for type-checking: init these variables with mock values out of the loop
        v_loss, pg_loss, old_approx_kl, approx_kl = (torch.zeros((1,)) for _ in range(4))
        for repeat in range(3):
            # shuffle the indices we use to access the data
            np.random.shuffle(b_index)
            for start in range(0, len(b_obs), batch_size):
                # select the indices we want to train on
                end = start + batch_size
                batch_index = b_index[start:end]

                _, newlogprob, entropy, value = agent.get_action_and_value_for_team(
                    b_obs[batch_index].transpose(0, 1),
                    b_actions.long()[batch_index].transpose(0, 1)
                )
                logratio = newlogprob - b_logprobs[batch_index]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clip_fracs += [
                        ((ratio - 1.0).abs() > clip_coef).float().mean().item()
                    ]

                # normalize advantaegs
                advantages = b_advantages[batch_index]
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

                # Policy loss
                pg_loss1 = -b_advantages[batch_index] * ratio
                pg_loss2 = -b_advantages[batch_index] * torch.clamp(
                    ratio, 1 - clip_coef, 1 + clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                value = value.flatten()
                v_loss_unclipped = (value - b_returns[batch_index]) ** 2
                v_clipped = b_values[batch_index] + torch.clamp(
                    value - b_values[batch_index],
                    -clip_coef,
                    clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[batch_index]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.floating(np.nan) if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        print(f"Training episode {episode}")
        print(f"Episodic Return: {np.mean(total_episodic_return)}")
        print(f"Episode Length: {end_step}")
        print("")
        print(f"Value Loss: {v_loss.item()}")
        print(f"Policy Loss: {pg_loss.item()}")
        print(f"Old Approx KL: {old_approx_kl.item()}")
        print(f"Approx KL: {approx_kl.item()}")
        print(f"Clip Fraction: {np.mean(clip_fracs)}")
        print(f"Explained Variance: {explained_var.item()}")
        print("\n-------------------------------------------\n")

    """ RENDER THE POLICY """
    env = cast(ParallelEnv, simple_tag_v3.parallel_env(
        num_good=1,  # Number of prey
        num_adversaries=num_agents,  # Number of predators
        num_obstacles=2,
        max_cycles=max_cycles,
        continuous_actions=False
    ))

    agent.eval()

    with torch.no_grad():
        # render 5 episodes out
        for episode in range(5):

            env_obs, infos = env.reset(seed=None)
            agent_ids, prey_agent, prey_id = get_agents_from_rest_env(env)
            obs = batchify_obs(env_obs, device, agent_ids)
            terms = [False]
            truncs = [False]
            while env.agents:
                actions, logprobs, _, values = agent.get_on_the_fly_action_and_value_for_team(obs, agent_ids)
                prey_action = prey_agent.get_action(env_obs[prey_id], prey_id)
                env_obs, rewards, terms, truncs, infos = env.step(
                    { prey_id: prey_action, **unbatchify(actions, agent_ids) }
                )
                obs = batchify_obs(env_obs, device, agent_ids)
                terms = [terms[a] for a in terms]
                truncs = [truncs[a] for a in truncs]
            env.close()
