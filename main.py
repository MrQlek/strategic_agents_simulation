from __future__ import annotations

from enum import Enum
import random
import math

class RAEAgentReportEntry:
    def __init__(self, suplier_number: int, receiver_number: int, reception_rate: float) -> None:
        self.suplier_number = suplier_number
        self.receiver_number = receiver_number
        self.reception_rate = reception_rate

class RAEAgentReport:
    def __init__(self) -> None:
        self._entries = []
    
    def add_entry(self, entry: RAEAgentReportEntry) -> None:
        self._entries.append(entry)

    @property
    def entries(self):
        return self._entries

class AgentMode(Enum):
    HONEST = 0
    STRATEGIC = 1

class AgentConfig():
    def __init__(self, start_trust: float, mode: AgentMode,
                    x:float, y:float, z:float,
                    expoA:float, expoG:float,
                    kmin:int, kmax:int) -> None:

        self.start_trust_level_V = start_trust
        self.mode = mode

        self.good_will_honest_x = x
        self.good_will_p_y = y
        self.good_will_r_z = z

        self.expoA = expoA
        self.expoG = expoG

        self.min_suppliers_number_kmin = kmin
        self.max_suppliers_number_kmax = kmax

class Agent:
    def __init__(self, number, config: AgentConfig):
        self.number = number
        self.config = config
        self.trust_level_V = config.start_trust_level_V

        self.new_reception_trust_R = 0

    def _randomize_A(self):
        return math.pow(random.random(), 1/self.config.expoA)

    def _count_honset_p(self, receiver: Agent) -> float:
        if receiver.trust_level_V >= 1 - self.config.good_will_honest_x:
            return self._randomize_A()
        return 0

    def _count_strategic_p(self, receiver: Agent) -> float:
        return min(self.config.good_will_p_y, self._count_honset_p(receiver))

    def do_service(self, receiver: Agent) -> float:
        if self.config.mode == AgentMode.HONEST:
            return self._count_honset_p(receiver)
        if self.config.mode == AgentMode.STRATEGIC and receiver.config.mode == AgentMode.STRATEGIC:
            return self._count_honset_p(receiver)
        if self.config.mode == AgentMode.STRATEGIC:
            return self._count_strategic_p(receiver)
        assert(0)


    def _randomize_G(self):
        return math.pow(random.random(), 1/self.config.expoG)

    def _count_service_reception_rate_honest(self, service_answer_P: float) -> float:
        if self.trust_level_V >= 1 - self.config.good_will_honest_x:
            return self._randomize_G()*service_answer_P
        return 0
    
    def _count_service_reception_rate_strategic(self, service_answer_P: float) -> float:
        return min(self.config.good_will_p_z, self._count_honset_p(service_answer_P))

    def _count_service_reception_rate_R(self, agent: Agent, service_answer_P: float) -> float:
        if self.config.mode == AgentMode.HONEST:
            return self._count_service_reception_rate_honest(service_answer_P)
        if self.config.mode == AgentMode.STRATEGIC and agent.config.mode == AgentMode.STRATEGIC:
            return self._count_service_reception_rate_honest(service_answer_P)
        if self.config.mode == AgentMode.STRATEGIC:
            return self._count_service_reception_rate_strategic(service_answer_P)

    def do_work(self, agents_list: list) -> RAEAgentReport:
        chosen_agents = random.sample(list(filter(lambda x: x.number != self.number, agents_list)),
                                      random.randint(self.config.min_suppliers_number_kmin,
                                                        self.config.max_suppliers_number_kmax))
        
        report = RAEAgentReport()
        for agent in chosen_agents:
            service_answer_P = agent.do_service(self)
            service_reception_rate = self._count_service_reception_rate_R(agent, service_answer_P)
            report.add_entry(RAEAgentReportEntry(agent.number, self.number, service_reception_rate))

        return report


class RAE:
    def __init__(self, honest_agents_number: int, all_agents_number: int, agent_config: AgentConfig, iterations_number: int) -> None:
        self.agents = RAE._create_agents(honest_agents_number, all_agents_number, agent_config)
        self.iterations_number = iterations_number

    def do_simulation(self):
        for iteration_number in range(self.iterations_number):
            print(iteration_number)
            self._iteration_action()

    def _iteration_action(self):
        reports_entries = []
        for agent in self.agents:
            reports_entries += agent.do_work(self.agents).entries
        self._count_new_reception_trust_R_base_on_reports(reports_entries)


    def _count_new_reception_trust_R_base_on_reports(self, reports_entries: list(RAEAgentReportEntry)):
        for agent in self.agents:
            regarding_entries = list(filter(lambda x: x.suplier_number == agent.number, reports_entries))
            sum = 0
            for entry in regarding_entries:
                sum += self.agents[entry.receiver_number].trust_level_V*entry.reception_rate 
            agent.new_reception_trust_R = sum/len(regarding_entries)
            


    @staticmethod
    def _create_agents(honest_agents_number: int, all_agents_number: int, agent_config: AgentConfig) -> list(Agent):
        agent_config.mode = AgentMode.HONEST
        honest_agents = [Agent(i, agent_config) for i in range(honest_agents_number)]
        agent_config.mode = AgentMode.STRATEGIC
        strategic_agents = [Agent(i + honest_agents_number, agent_config) for i in range(all_agents_number - honest_agents_number)]
        return honest_agents + strategic_agents








def main():
    agent_config = AgentConfig(0.5, AgentMode.HONEST, 1, 1, 1, 1, 1, 50, 150)
    all_agents_number = 1000
    strategic_agents_number = 250
    honest_agents_number = all_agents_number - strategic_agents_number
    iterations_number = 100

    rae = RAE(honest_agents_number, all_agents_number, agent_config, iterations_number)
    rae.do_simulation()
    

if __name__ == "__main__":
    main()