from __future__ import annotations

from enum import Enum
import random
import math
from copy import copy

from sklearn.cluster import KMeans
import numpy
import matplotlib.pyplot as plt

cumulative_distribution_A = [6, 15, 26, 36, 37, 55, 44, 56, 47, 67, 66, 67, 75, 64, 72, 75, 73, 77, 85, 125, 116, 117, 116, 112, 127, 145, 101, 123, 120, 150, 162, 119, 125, 129, 137, 115, 148, 143, 150, 144, 134, 151, 139, 115, 158, 124, 120, 136, 134, 134, 117, 124, 131, 130, 141, 143, 118, 115, 131, 163, 119, 103, 125, 125, 124, 139, 117, 115, 109, 117, 108, 98, 117, 94, 120, 116, 127, 120, 120, 153, 142, 120, 138, 127, 125, 116, 121, 108, 156, 132, 122, 108, 130, 119, 123, 117, 132, 114, 626, 167568]
cumulative_distribution_A_choice = []
for idx, item in enumerate(cumulative_distribution_A):
    cumulative_distribution_A_choice+=[(idx+1)]*item

cumulative_distribution_G = [2, 22, 29, 0, 30, 1, 0, 33, 29, 0, 1, 0, 14, 8, 11, 5, 34, 64, 45, 11, 83, 146, 101, 100, 112, 161, 192, 105, 109, 208, 198, 157, 87, 197, 222, 184, 586, 158, 172, 215, 437, 248, 214, 161, 351, 298, 417, 386, 443, 728, 566, 570, 559, 501, 1633, 586, 622, 841, 1002, 1844, 7236, 1304, 1604, 2590, 5582, 4274, 2462, 2338, 4840, 2947, 3377, 3791, 4648, 3680, 3319, 3522, 5746, 4503, 4025, 4588, 3652, 4450, 5862, 4968, 4739, 4196, 5498, 11824, 4673, 5943, 4306, 5218, 4748, 3565, 1379, 1876, 3155, 1752, 1259, 824, 94]
cumulative_distribution_G_choice = []
for idx, item in enumerate(cumulative_distribution_G):
    cumulative_distribution_G_choice+=[(idx)]*item



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

    def _randomize_A_partA(self):
        return math.pow(random.random(), 1/self.config.expoA)

    def _randomize_A_partB(self):
        return random.choice(cumulative_distribution_A_choice)/100

    def _randomize_A(self):
        return self._randomize_A_partB()

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

    def _randomize_G_partA(self):
        return math.pow(random.random(), 1/self.config.expoG)

    def _randomize_G_partB(self):
        return random.choice(cumulative_distribution_G_choice)/100

    def _randomize_G(self):
        return self._randomize_G_partB()

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
        print(low_set_trust_level)
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

        print(f"{low_set_size=}")
        print(f"{low_set_sum=}")
        print(f"{high_set_size=}")
        print(f"{high_set_sum=}")

        return (low_set_sum/low_set_size)/(high_set_sum/high_set_size)


    @staticmethod
    def _create_agents(honest_agents_number: int, all_agents_number: int, agent_config: AgentConfig) -> list(Agent):
        agent_config.mode = AgentMode.HONEST
        honest_agents = [Agent(i, agent_config) for i in range(honest_agents_number)]
        agent_config.mode = AgentMode.STRATEGIC
        strategic_agents = [Agent(i + honest_agents_number, agent_config) for i in range(all_agents_number - honest_agents_number)]
        return honest_agents + strategic_agents


class SimulationConfig():
    def __init__(self) -> None:
        self.start_trust = 1
        self.x = float(input("Parametr x: "))
        self.y = float(input("Parametr y: "))
        self.z = float(input("Parametr z: "))
        self.expoA = 1
        self.expoG = 1
        self.kmin = 50
        self.kmax = 150
        # self.all_agents_number = 200
        # self.strategic_agents_number = 50
        self.all_agents_number = 1000
        self._strategic_agents_number = int(input("Ilosc agentow strategicznych w systemie (liczba wszystkich agentow wysnosi 1000: "))
        assert(self._strategic_agents_number < self.all_agents_number)
        self.honest_agents_number = self.all_agents_number - self.strategic_agents_number
        self.iterations_number = int(input("Liczba symulowanych iteracji: "))

    @property
    def strategic_agents_number(self):
        return self._strategic_agents_number
    
    @strategic_agents_number.setter
    def strategic_agents_number(self, val):
        if val > self.all_agents_number:
            return
        self._strategic_agents_number = val
        self.honest_agents_number = self.all_agents_number - self.strategic_agents_number

    @property
    def agent_config(self):
        return AgentConfig(
            self.start_trust,
            AgentMode.HONEST,
            self.x, self.y, self.z,
            self.expoA, self.expoG,
            self.kmin, self.kmax)

    def __str__(self):
        return f"PARAMS: N={self.all_agents_number}, S={self.strategic_agents_number}, x={self.x:.1f}, y={self.y:.1f}, z={self.z:.1f}"


class SimulationResultsOutput():
    def __init__(self, result: SimulationResults, simulation_config: SimulationConfig) -> None:
        self.result = result
        self.simulation_config = simulation_config

    def _draw_trust_trajectory_chart(self):
        assert(len(self.result.avg_honest_agents_trust) == len(self.result.avg_strategic_agents_trust))
        x_series = [i for i in range(len(self.result.avg_honest_agents_trust))]

        honest_agents_trust_func, = plt.plot(x_series, self.result.avg_honest_agents_trust, label="honest agents")
        strategic_agents_trust_func, = plt.plot(x_series, self.result.avg_strategic_agents_trust, label="strategic agents")
        plt.xlabel("Iterations")
        plt.ylabel("Trust trajectory")
        plt.gca().set_ylim([0,1.1])
        plt.title(self.simulation_config.__str__())
        plt.legend(handles=[honest_agents_trust_func, strategic_agents_trust_func])
        plt.grid()

    def _draw_service_influence_chart(self):
        x_series = [i for i in range(len(self.result.avg_honest_services_influence_on_strategic_agents_F))]

        serive_influence, = plt.plot(x_series, self.result.avg_honest_services_influence_on_strategic_agents_F, label="honest agents")
        plt.xlabel("Iterations")
        plt.ylabel("Honest agents service influence on strategic agents")
        plt.gca().set_ylim([-1,1])
        plt.title(self.simulation_config.__str__())
        plt.legend(handles=[serive_influence])
        plt.grid()


    def _save_current_chart_to_file(self, file_name: str):
        plt.savefig(f'./results/{file_name}.png')
        plt.close()


    def save_charts_to_files(self):
        self._draw_trust_trajectory_chart()
        self._save_current_chart_to_file(f"trust_{self.simulation_config.__str__().replace(', ', '_').replace(': ', '_')}")
        self._draw_service_influence_chart()
        self._save_current_chart_to_file(f"influence_{self.simulation_config.__str__().replace(', ', '_').replace(': ', '_')}")
    
    def display_charts(self):
        self._draw_trust_trajectory_chart()
        plt.show()
        pass

    def save_result_to_csv(self):
        assert(len(self.result.avg_honest_agents_trust) == len(self.result.avg_strategic_agents_trust))
        assert(len(self.result.avg_honest_agents_trust) == len(self.result.avg_honest_services_influence_on_strategic_agents_F))
        with open(f"results/result_{self.simulation_config.__str__().replace(', ', '_').replace(': ', '_')}.csv", 'w') as f:
            f.write("iteration;avg_honest_agents_trust;avg_strategic_agents_trust;avg_honest_services_influence_on_strategic_agents_F\r\n")
            for i in range(len(self.result.avg_honest_agents_trust)):
                f.write(f"{i};{self.result.avg_honest_agents_trust[i]};{self.result.avg_strategic_agents_trust[i]};{self.result.avg_honest_services_influence_on_strategic_agents_F[i]}\r\n")

def make_simulation(simulation_config: SimulationConfig):
    rae = RAE(simulation_config.honest_agents_number,
                simulation_config.all_agents_number,
                simulation_config.agent_config,
                simulation_config.iterations_number)
    result = rae.do_simulation()
    result_output = SimulationResultsOutput(result, simulation_config)
    result_output.save_charts_to_files()
    result_output.save_result_to_csv()


def main():
    simulation_config = SimulationConfig()
    print(simulation_config.__str__())
    make_simulation(simulation_config)

if __name__ == "__main__":
    main()
