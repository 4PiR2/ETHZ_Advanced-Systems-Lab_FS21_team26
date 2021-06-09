import sys
import os
import matplotlib.pyplot as plt
import pprint
import pandas as pd

main_file = "./src/main.cpp"

def get_timers(raw, query):
    lines = [line[6:].strip().split(",") for line in raw.split("\n") if line.startswith("TIMER:")]
    lines = {line[0]: float(line[2]) for line in lines}
    return lines.get(query, -1)

def get_loops(raw):
    lines = [line[6:].strip() for line in raw.split("\n") if line.startswith("LOOPS:")]
    return int(lines[0])
    
def work(nlist, flag, macro, function_label, flop_func):
    command_base = "g++ -o main -DSYM_AFF_COUNT_LOOPS -MMD -MP -ffast-math -march=native {} {} ./src/main.cpp && ./main {}"
    rt, flops = [], []
    for n in nlist:
        print(n)
        command = command_base.format(flag, macro, n)
        raw = os.popen(command).read()
        timer = get_timers(raw, function_label)
        flops.append(flop_func(n) * get_loops(raw))
        rt.append(timer)

    return {
        "n": nlist,
        "flops": flops,
        "cycles": rt
    }

def plot_performance(data):
    fig, ax = plt.subplots(figsize=(10,8))
    # fig.delaxes(axs[1,1])
    for name, v in data.items():
        Linewidths=[2, 2, 2]

        xs = [str(n) for n in v["n"]]
        ys = [v["flops"][i] / v["cycles"][i] for i in range(len(xs))]

        ax.set_title(r"Performance [flops/cycle]", loc="left", fontsize=20,pad=16)

        # grid
        ax.set_facecolor("lavender")
        for b in ax.get_ygridlines():
            b.set_color('white')
            b.set_linewidth(2)
        ax.tick_params(axis='both', top=False, bottom=True, left=False, right=False, direction='out', which='both', labelsize=14,pad=8)

        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.grid(True, axis="y")
        ax.set_xlabel(r'n')
        ax.spines['bottom'].set_linewidth(2)
        ax.xaxis.set_tick_params(width=2)

        ax.plot(xs, ys, linewidth=Linewidths[0], label=name)

        ax.legend(fontsize="large")

    fig.tight_layout()
    plt.savefig("output/performance.png")


def plot_runtime(data):
    fig, ax = plt.subplots(figsize=(10,8))
    # fig.delaxes(axs[1,1])
    for name, v in data.items():
        Linewidths=[2, 2, 2]

        xs = [str(n) for n in v["n"]]
        ys = [v["cycles"][i] for i in range(len(xs))]

        ax.set_title(r"Runtime [cycle]", loc="left", fontsize=20,pad=16)

        # grid
        ax.set_facecolor("lavender")
        for b in ax.get_ygridlines():
            b.set_color('white')
            b.set_linewidth(2)
        ax.tick_params(axis='both', top=False, bottom=True, left=False, right=False, direction='out', which='both', labelsize=14,pad=8)

        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.grid(True, axis="y")
        ax.set_xlabel(r'n')
        ax.spines['bottom'].set_linewidth(2)
        ax.xaxis.set_tick_params(width=2)

        ax.plot(xs, ys, linewidth=Linewidths[0], label=name)

        ax.legend(fontsize="large")

    fig.tight_layout()
    plt.savefig("output/runtime.png")


def plot_speedup(data):
    fig, ax = plt.subplots(figsize=(10,8))
    # fig.delaxes(axs[1,1])
    for name, v in data.items():
        Linewidths=[2, 2, 2]

        xs = [str(n) for n in v["n"]]
        ys = [data["vec-base"]["cycles"][i] / v["cycles"][i] for i in range(len(xs))]

        ax.set_title(r"Speedup [x]", loc="left", fontsize=20,pad=16)

        # grid
        ax.set_facecolor("lavender")
        for b in ax.get_ygridlines():
            b.set_color('white')
            b.set_linewidth(2)
        ax.tick_params(axis='both', top=False, bottom=True, left=False, right=False, direction='out', which='both', labelsize=14,pad=8)

        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.grid(True, axis="y")
        ax.set_xlabel(r'n')
        ax.spines['bottom'].set_linewidth(2)
        ax.xaxis.set_tick_params(width=2)

        ax.plot(xs, ys, linewidth=Linewidths[0], label=name)

        ax.legend(fontsize="large")

    fig.tight_layout()
    plt.savefig("output/speedup.png")

def main():
    flags = {
        # 'nvec': '-O3 -fno-tree-vectorize',
        'vec': '-O3'
    }
    macros = {
        'avx2': '-DV_SCALAR -DSYM_AFF_PA_AVX2',
        'scalar': '-DV_SCALAR -DSYM_AFF_PA_SCALAR_UP1',
        'base': '-DV_BASELINE'
    }
    flop_funcs = {
        'avx2': lambda n: 10 * n,
        'scalar': lambda n: 11 * n,
        'base': lambda n: 20 * n
    }
    nlist = [100 * i for i in range(1, 10)] + [1000 * i for i in range(1, 11)]
    # nlist = [100, 200, 300, 400, 500]
    
    data = {}
    for kf, vf in flags.items():
        for km, vm in macros.items():
            key = kf + '-' + km
            print(key, "is running...")
            data[key] = work(nlist, vf, vm, "PA", flop_funcs[km])
            df = pd.DataFrame(data[key])
            csvfn = "./output/" + key + ".csv"
            df.to_csv(csvfn, index=None)
            # print(df)

    plot_performance(data)
    plot_runtime(data)
    plot_speedup(data)

    

if __name__ == "__main__":
    main()
            

