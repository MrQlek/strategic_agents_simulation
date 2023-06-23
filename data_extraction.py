from collections import Counter
import matplotlib.pyplot as plt

def save_current_chart_to_file(file_name: str):
    plt.savefig(f'./{file_name}.png')
    plt.close()

def get_lines():
    lines = []
    with open("LoanData_Bondora.csv", "r") as f:
        lines = f.readlines()

    return lines

def get_service_quality_list(lines):
    header = lines[0].split(",")
    data = lines[1:]

    amount_index = header.index("\"Amount\"")
    applied_amount_index = header.index("\"AppliedAmount\"")

    result = []
    for line in data:
        fields = line.split(",")
        amount = float(fields[amount_index].replace("\"", ""))
        applied_amount = float(fields[applied_amount_index].replace("\"", ""))

        result.append(amount/applied_amount)

    return result

class Probability():
    def __init__(self, quality, occurance):
        self.quality = quality
        self.occurance = occurance

    def __lt__(self, other):
        return self.quality < other.quality

    def __repr__(self):
        return f"{self.quality}: {self.occurance}"

def get_service_probability(service_quality):
    new_list = []
    for val in service_quality:
        new_list.append(int(val*100))
    res = []
    for key, val in Counter(new_list).items():
        res.append(Probability(key, val))

    res.sort()
    print(res)

    x_series = [item.quality for item in res]
    y_series = []
    _sum = sum([item.occurance for item in res])
    for item in res:
        if(len(y_series) == 0):
            y_series.append(item.occurance/_sum)
        else:
            y_series.append((item.occurance/_sum) + y_series[-1])

    print(y_series[-2])

    plt.gca().set_ylim([0,0.1])
    plt.gca().set_xlim([0,110])
    service_probability_chart, = plt.plot(x_series, y_series)
    save_current_chart_to_file("service_quality")

    plt.gca().set_ylim([0,1.1])
    plt.gca().set_xlim([0,110])
    service_probability_chart, = plt.plot(x_series, y_series)
    save_current_chart_to_file("service_quality2")


def get_report_quality_list(lines):
    header = lines[0].split(",")
    data = lines[1:]

    default_index = header.index("\"ProbabilityOfDefault\"")

    result = []
    for line in data:
        fields = line.split(",")
        try:
            default = float(fields[default_index].replace("\"", ""))
        except ValueError:
            continue

        result.append(1 - default)

    return result


def get_report_probability(report_quality):
    new_list = []
    for val in report_quality:
        new_list.append(int(val*100))
    res = []
    for key, val in Counter(new_list).items():
        res.append(Probability(key, val))

    res.sort()
    print(res)

    x_series = [item.quality for item in res]
    y_series = []
    _sum = sum([item.occurance for item in res])
    for item in res:
        if(len(y_series) == 0):
            y_series.append(item.occurance/_sum)
        else:
            y_series.append((item.occurance/_sum) + y_series[-1])

    print(y_series[-2])

    plt.gca().set_ylim([0,0.1])
    plt.gca().set_xlim([0,110])
    service_probability_chart, = plt.plot(x_series, y_series)
    save_current_chart_to_file("report_quality")

    plt.gca().set_ylim([0,1.1])
    plt.gca().set_xlim([0,110])
    service_probability_chart, = plt.plot(x_series, y_series)
    save_current_chart_to_file("report_quality2")





def main():
    lines = get_lines()
    print(lines[0])
    service_quality = get_service_quality_list(lines)

    get_service_probability(service_quality)

    report_quality = get_report_quality_list(lines)

    get_report_probability(report_quality)


if __name__ == "__main__":
    main()
