
def get_action_value(mdp, state_values, state, action, gamma):
    """ Computes Q(s,a) as in formula above """

    q = 0
    for state_ in mdp.get_next_states(state, action):
        prob = mdp.get_transition_prob(state, action, state_)
        reward = mdp.get_reward(state, action, state_)
        q += prob * (reward + (gamma*state_values[state_]))

    return q
