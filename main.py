from enum import Enum
import random
import math

class AgentMode(Enum):
    HONEST = 0
    STRATEGIC = 1

class Agent:
    def __init__(self):
        self.trust_level_V = 0
        self.mode = AgentMode.HONEST

        self.good_will_p_y = 0
        self.good_will_r_z = 0
        self.good_will_honest_x = 0

        self.expoA = 0
        self.expoG = 0

        self.min_suppliers_number_kmin = 0
        self.max_suppliers_number_kmax = 0

        self.new_reception_trust_R = 0

    def do_service(self, receiver: Agent) -> float:
        if self.mode == AgentMode.HONEST:
            return self._count_honset_p(receiver)
        if self.mode == AgentMode.STRATEGIC and receiver.mode == AgentMode.STRATEGIC:
            return self._count_honset_p(receiver)
        if self.mode == AgentMode.STRATEGIC:
            return self._count_strategic_p(receiver)
        assert(0)

    def _count_honset_p(self, receiver: Agent) -> float:
        if receiver.trust_level_V >= 1 - self.good_will_honest_x:
            return self._randomize_A()
        return 0

    def _count_strategic_p(self, receiver: Agent) -> float:
        return min(self.good_will_p_y, self._count_honset_p(receiver))


    def _randomize_A(self):
        math.pow(random.random(), 1/self.expoA)

    def _randomize_G(self):
        math.pow(random.random(), 1/self.expoG)

    def do_work(self, agents_list: list):
        chosen_agents = random.sample(agents_list,
                                      random.randint(self.min_suppliers_number_kmin, self.max_suppliers_number_kmax))
        
        reception_trust_R_nominator = 0
        for agent in chosen_agents:
            service_answer_P = agent.do_service(self)
            service_reception = self._count_service_reception_rate_R(self, agent, service_answer_P)
            reception_trust_R_nominator += self.trust_level_V*service_reception
        self.new_reception_trust_R = reception_trust_R_nominator/len(chosen_agents)


    def _count_service_reception_rate_R(self, agent: Agent, service_answer_P: float) -> float:
        if self.mode == AgentMode.HONEST:
            return self._count_service_reception_rate_honest(service_answer_P)
        if self.mode == AgentMode.STRATEGIC and agent.mode == AgentMode.STRATEGIC:
            return self._count_service_reception_rate_honest(service_answer_P)
        if self.mode == AgentMode.STRATEGIC:
            return self._count_service_reception_rate_strategic(service_answer_P)
        
    def _count_service_reception_rate_honest(self, service_answer_P: float) -> float:
        if self.trust_level_V >= 1 - self.good_will_honest_x:
            return self._randomize_G()*service_answer_P
        return 0
    
    def _count_service_reception_rate_strategic(self, service_answer_P: float) -> float:
        return min(self.good_will_p_z, self._count_honset_p(service_answer_P))








def main():
    pass

if __name__ == "__main__":
    main()