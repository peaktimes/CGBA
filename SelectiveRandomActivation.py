import random

import mesa
from mesa.model import Model
from mesa.agent import Agent
from typing import Dict, Iterator, List, Type, Union


class SelectiveRandomActivation(mesa.time.RandomActivation):


    def agent_buffer(self, shuffled: bool = False) -> Iterator[Agent]:
        """Simple generator that yields the agents while letting the user
        remove and/or add agents during stepping.
        """
        randomScale = self.model.numAgents
        agent_keys = list(self._agents.keys())
        selectedAgentKeys = agent_keys[:randomScale]
        if shuffled:
            self.model.random.shuffle(selectedAgentKeys)
            agent_keys[:randomScale] = selectedAgentKeys
        for key in agent_keys:
            if key in self._agents:
                yield self._agents[key]
