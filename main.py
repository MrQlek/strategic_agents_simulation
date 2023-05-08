from __future__ import annotations

from enum import Enum
import random
import math
from copy import copy

from sklearn.cluster import KMeans
import numpy

class RAEAgentReportEntry:
    def __init__(self, suplier_number: int, receiver_number: int,
                    reception_rate: float, service_answer_P: float) -> None:
        self.suplier_number = suplier_number
        self.receiver_number = receiver_number
        self.reception_rate = reception_rate
        self.service_answer_P = service_answer_P

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
        self.config = copy(config)
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
        return min(self.config.good_will_r_z, self._count_service_reception_rate_honest(service_answer_P))

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
            report.add_entry(RAEAgentReportEntry(agent.number, self.number,
                                                    service_reception_rate, service_answer_P))

        return report

class SimulationResults():
    def __init__(self) -> None:
        self.avg_strategic_agents_trust = []
        self.avg_honest_agents_trust = []
        self.avg_honest_services_influence_on_strategic_agents_F = []

    def count_statistics(self, agents_list: list(Agent), reports_entries: list(RAEAgentReportEntry)):
        honest_agents_trust_sum = 0
        honest_agents_number = 0
        strategic_agents_trust_sum = 0
        strategic_agents_number = 0

        for agent in agents_list:
            if agent.config.mode == AgentMode.HONEST:
                honest_agents_trust_sum += agent.trust_level_V
                honest_agents_number += 1
            elif agent.config.mode == AgentMode.STRATEGIC:
                strategic_agents_trust_sum += agent.trust_level_V
                strategic_agents_number += 1
            else:
                assert(0)

        avg_F = self._count_avg_F(agents_list, reports_entries)
        self._add_iteration_values(strategic_agents_trust_sum/strategic_agents_number,
                                    honest_agents_trust_sum/honest_agents_number,
                                    avg_F)

    def _count_avg_F(self, agents_list: list(Agent), reports_entries: list(RAEAgentReportEntry)):
        service_answer_when_honest_provide_to_strategic_sum = 0
        service_answer_when_strategic_provider_to_honest_sum = 0

        number_of_honest_provide_to_strategic = 0
        number_of_strategic_provide_to_honest = 0

        for report_entry in reports_entries:
            if agents_list[report_entry.suplier_number].config.mode == AgentMode.HONEST \
                and agents_list[report_entry.receiver_number].config.mode == AgentMode.STRATEGIC:

                service_answer_when_honest_provide_to_strategic_sum += report_entry.service_answer_P
                number_of_honest_provide_to_strategic += 1
            elif agents_list[report_entry.suplier_number].config.mode == AgentMode.STRATEGIC \
                and agents_list[report_entry.receiver_number].config.mode == AgentMode.HONEST:

                service_answer_when_strategic_provider_to_honest_sum += report_entry.service_answer_P
                number_of_strategic_provide_to_honest += 1

        return (service_answer_when_honest_provide_to_strategic_sum/number_of_honest_provide_to_strategic) - (service_answer_when_strategic_provider_to_honest_sum/number_of_strategic_provide_to_honest)

                     
    def _add_iteration_values(self, avg_strategic_agents_trus: float,
                                avg_honest_agents_trust: float,
                                avg_F: float) -> None:
        self.avg_strategic_agents_trust.append(avg_strategic_agents_trus)
        self.avg_honest_agents_trust.append(avg_honest_agents_trust)
        self.avg_honest_services_influence_on_strategic_agents_F.append(avg_F)

class RAE:
    def __init__(self, honest_agents_number: int, all_agents_number: int, agent_config: AgentConfig, iterations_number: int) -> None:
        self.agents = RAE._create_agents(honest_agents_number, all_agents_number, agent_config)
        self.iterations_number = iterations_number

    def do_simulation(self) -> SimulationResults:
        simulation_result = SimulationResults()
        for iteration_number in range(self.iterations_number):
            print(iteration_number)
            self._iteration_action(simulation_result)
        return simulation_result

    def _iteration_action(self, simulation_result: SimulationResults):
        reports_entries = []
        for agent in self.agents:
            reports_entries += agent.do_work(self.agents).entries
        self._count_new_reception_trust_R_base_on_reports(reports_entries)
        self._assigne_new_trust_levels()
        simulation_result.count_statistics(self.agents, reports_entries)


    def _count_new_reception_trust_R_base_on_reports(self, reports_entries: list(RAEAgentReportEntry)):
        for agent in self.agents:
            regarding_entries = list(filter(lambda x: x.suplier_number == agent.number, reports_entries))
            sum = 0
            for entry in regarding_entries:
                sum += self.agents[entry.receiver_number].trust_level_V*entry.reception_rate 
            agent.new_reception_trust_R = sum/len(regarding_entries)

    def _assigne_new_trust_levels(self):
        clusters_boundry = self._get_clusters_boundry()
        low_set_trust_level = self._count_low_set_trust_level(clusters_boundry)
        for agent in self.agents:
            if agent.new_reception_trust_R < clusters_boundry:
                agent.trust_level_V = low_set_trust_level
            else:
                agent.trust_level_V = 1


    def _get_clusters_boundry(self):
        reception_trust_R = []
        for agent in self.agents:
            reception_trust_R.append(agent.new_reception_trust_R)

        reception_trust_R_array = numpy.array(reception_trust_R).reshape(-1,1)
        kmeans = KMeans(n_clusters = 2, n_init='auto')
        kmeans.fit(reception_trust_R_array)
        print(kmeans.cluster_centers_)
        return (kmeans.cluster_centers_[0][0] + kmeans.cluster_centers_[1][0])/2

    def _count_low_set_trust_level(self, clusters_boundry: float) -> float:
        high_set_sum = 0
        high_set_size = 0

        low_set_sum = 0
        low_set_size = 0
        for agent in self.agents:
            if agent.new_reception_trust_R < clusters_boundry:
                low_set_sum += agent.new_reception_trust_R 
                low_set_size += 1
            else:
                high_set_sum += agent.new_reception_trust_R 
                high_set_size += 1

        return (low_set_sum/low_set_size)/(high_set_sum/high_set_size)


    @staticmethod
    def _create_agents(honest_agents_number: int, all_agents_number: int, agent_config: AgentConfig) -> list(Agent):
        agent_config.mode = AgentMode.HONEST
        honest_agents = [Agent(i, agent_config) for i in range(honest_agents_number)]
        agent_config.mode = AgentMode.STRATEGIC
        strategic_agents = [Agent(i + honest_agents_number, agent_config) for i in range(all_agents_number - honest_agents_number)]
        return honest_agents + strategic_agents








def main():
    agent_config = AgentConfig(1, AgentMode.HONEST, 0.5, 0.8, 0.3, 1, 1, 50, 150)
    all_agents_number = 200
    strategic_agents_number = 50
    # all_agents_number = 1000
    # strategic_agents_number = 250
    honest_agents_number = all_agents_number - strategic_agents_number
    iterations_number = 10

    rae = RAE(honest_agents_number, all_agents_number, agent_config, iterations_number)
    result = rae.do_simulation()
    print(result.avg_honest_agents_trust)
    print(result.avg_strategic_agents_trust)
    print(result.avg_honest_services_influence_on_strategic_agents_F)


if __name__ == "__main__":
    main()