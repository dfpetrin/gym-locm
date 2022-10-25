import gym
import numpy as np

from gym_locm.agents import RandomDraftAgent, RandomBattleAgent
from gym_locm.engine import Phase, Action, PlayerOrder
from gym_locm.envs.base_env import LOCMEnv
from gym_locm.exceptions import GameIsEndedError, MalformedActionError


class LOCMBattleEnv(LOCMEnv):
    metadata = {"render.modes": ["text", "native"]}

    def __init__(
        self,
        draft_agents=(RandomDraftAgent(), RandomDraftAgent()),
        return_action_mask=False,
        seed=None,
        items=True,
        k=3,
        n=30,
    ):
        super().__init__(seed=seed, items=items, k=k, n=n)

        self.rewards = [0.0]

        self.draft_agents = draft_agents

        for draft_agent in self.draft_agents:
            draft_agent.reset()
            draft_agent.seed(seed)

        self.return_action_mask = return_action_mask

        player_features = 4  # hp, mana, next_rune, next_draw
        cards_in_hand = 8
        card_features = 16 if self.items else 12
        friendly_cards_on_board = 6
        friendly_board_card_features = 9
        enemy_cards_on_board = 6
        enemy_board_card_features = 8

        # 238 features if using items else 206 features
        self.state_shape = (
            player_features * 2
            + cards_in_hand * card_features
            + friendly_cards_on_board * friendly_board_card_features
            + enemy_cards_on_board * enemy_board_card_features
        )
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.state_shape,), dtype=np.float32
        )

        if self.items:
            # 145 possible actions
            self.action_space = gym.spaces.Discrete(145)
        else:
            # 41 possible actions
            self.action_space = gym.spaces.Discrete(41)

        self.reward_range = (-1, 1)

        # play through draft
        while self.state.phase == Phase.DRAFT:
            for agent in self.draft_agents:
                action = agent.act(self.state)

                self.state.act(action)

    def step(self, action):
        """Makes an action in the game."""
        # if the battle is finished, there should be no more actions
        if self._battle_is_finished:
            raise GameIsEndedError()

        # check if an action object or an integer was passed
        if not isinstance(action, Action):
            try:
                action = int(action)
            except ValueError:
                error = (
                    f"Action should be an action object "
                    f"or an integer, not {type(action)}"
                )

                raise MalformedActionError(error)

            action = self.decode_action(action)

        # less property accesses
        state = self.state

        # execute the action
        if action is not None:
            state.act(action)
        else:
            state.was_last_action_invalid = True

        # build return info
        winner = state.winner

        reward = 0
        done = winner is not None
        info = {
            "phase": state.phase,
            "turn": state.turn,
            "winner": winner,
            "invalid": state.was_last_action_invalid,
        }

        if self.return_action_mask:
            info["action_mask"] = self.state.action_mask

        if winner is not None:
            reward = 1 if winner == PlayerOrder.FIRST else -1

            self.rewards[-1] += reward

        return self.encode_state(), reward, done, info

    def reset(self) -> np.array:
        """
        Resets the environment.
        The game is put into its initial state and all agents are reset.
        """
        # reset the state
        super().reset()

        # reset all agents' internal state
        for agent in self.draft_agents:
            agent.reset()
            agent.seed(self._seed)

        # play through draft
        while self.state.phase == Phase.DRAFT:
            for agent in self.draft_agents:
                self.state.act(agent.act(self.state))

        self.rewards.append(0.0)

        return self.encode_state()

    def _encode_state_draft(self):
        pass

    def _encode_state_battle(self):
        encoded_state = np.full(self.state_shape, 0, dtype=np.float32)

        p0, p1 = self.state.current_player, self.state.opposing_player

        def fill_cards(card_list, up_to, features):
            remaining_cards = up_to - len(card_list)

            return card_list + [[0] * features for _ in range(remaining_cards)]

        all_cards = []

        # convert all cards in hand to features
        hand = list(map(self.encode_card, p0.hand))

        # if not using items, clip card type features
        if not self.items:
            hand = list(map(lambda c: c[4:], hand))

        # add dummy cards up to the card limit
        hand = fill_cards(hand, up_to=8, features=16 if self.items else 12)

        # add to card list
        all_cards.extend([feature for card in hand for feature in card])

        # in current player's lanes
        for location in (p0.lanes[0], p0.lanes[1]):
            # convert all cards to features
            location = list(map(self.encode_friendly_card_on_board, location))

            # add dummy cards up to the card limit
            location = fill_cards(location, up_to=3, features=9)

            # add to card list
            all_cards.extend([feature for card in location for feature in card])

        # in opposing player's lanes
        for location in (p1.lanes[0], p1.lanes[1]):
            # convert all cards to features
            location = list(map(self.encode_enemy_card_on_board, location))

            # add dummy cards up to the card limit
            location = fill_cards(location, up_to=3, features=8)

            # add to card list
            all_cards.extend([feature for card in location for feature in card])

        # players info
        encoded_state[:8] = self.encode_players(p0, p1)
        encoded_state[8:] = np.array(all_cards).flatten()

        return encoded_state

    def get_episode_rewards(self):
        return self.rewards


class LOCMBattleSingleEnv(LOCMBattleEnv):
    def __init__(self, battle_agent=RandomBattleAgent(), play_first=True, **kwargs):
        # init the env
        super().__init__(**kwargs)

        # also init the battle agent and the new parameter
        self.battle_agent = battle_agent
        self.play_first = play_first

        # reset the battle agent
        self.battle_agent.reset()

    def reset(self) -> np.array:
        """
        Resets the environment.
        The game is put into its initial state and all agents are reset.
        """
        # reset what is needed
        encoded_state = super().reset()

        # also reset the battle agent
        self.battle_agent.reset()

        # if playing second, have first player play
        if not self.play_first:
            while self.state.current_player.id != PlayerOrder.SECOND:
                super().step(self.battle_agent.act(self.state))

        return encoded_state

    def step(self, action):
        """Makes an action in the game."""
        player = self.state.current_player.id

        # do the action
        state, reward, done, info = super().step(action)

        was_invalid = info["invalid"]

        # have opponent play until its player's turn or there's a winner
        while self.state.current_player.id != player and self.state.winner is None:
            action = self.battle_agent.act(self.state)

            state, reward, done, info = super().step(action)

            if info["invalid"] and not done:
                state, reward, done, info = super().step(0)
                break

        info["invalid"] = was_invalid

        if not self.play_first:
            reward = -reward

        return state, reward, done, info


class LOCMBattleSelfPlayEnv(LOCMBattleEnv):
    def __init__(self, play_first=True, adversary_policy=None, **kwargs):
        # init the env
        super().__init__(**kwargs)

        # also init the new parameters
        self.play_first = play_first
        self.adversary_policy = adversary_policy

    def reset(self) -> np.array:
        """
        Resets the environment.
        The game is put into its initial state and all agents are reset.
        """
        # reset what is needed
        encoded_state = super().reset()

        # also reset the battle agent
        self.play_first = not self.play_first

        # if playing second, have first player play
        if not self.play_first:
            while self.state.current_player.id != PlayerOrder.SECOND:
                state = self.encode_state()
                action = self.adversary_policy(state)

                state, reward, done, info = super().step(action)

                if info["invalid"] and not done:
                    state, reward, done, info = super().step(0)
                    break

        return encoded_state

    def step(self, action):
        """Makes an action in the game."""
        player = self.state.current_player.id

        # do the action
        state, reward, done, info = super().step(action)

        was_invalid = info["invalid"]

        # have opponent play until its player's turn or there's a winner
        while self.state.current_player.id != player and self.state.winner is None:
            state = self.encode_state()
            action = self.adversary_policy(state)

            state, reward, done, info = super().step(action)

            if info["invalid"] and not done:
                state, reward, done, info = super().step(0)
                break

        info["invalid"] = was_invalid

        if not self.play_first:
            reward = -reward

        return state, reward, done, info
