import subprocess
import matplotlib.pyplot as plt
import re


def main():

    shot_o = 10
    random_params_o = 5
    possible_outcome_o = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    input_sizes_o = []
    runtimes_o = [2.601216, 3.035040, 1.958048,
                  2.277024, 4.068320, 4.841248, 8.637120, 13.143296, 23.798368, 37.499680]

    for i in range(len(runtimes_o)):
        input_sizes_o.append((possible_outcome_o[i] ** random_params_o)*shot_o)

    plt.figure()
    plt.plot(input_sizes_o, runtimes_o, marker='o')
    for i, (input_size, runtime) in enumerate(zip(input_sizes_o, runtimes_o)):
        plt.text(input_size, runtime,
                 f'({input_size}, {runtime:.2f})', ha='right', va='bottom')

    plt.xlabel("Input size")
    plt.ylabel("Runtime (ms)")
    plt.title("Runtime vs Input size")
    plt.grid(True)

    # max_runtime = max(runtimes_o)
    # max_input_size = max(input_sizes_o)

    # x_ticks = [i**5 for i in range(1, int(max_input_size**(1/5)) + 2)]  # calculate up to the next integer
    # y_ticks = [i**5 for i in range(1, int(max_runtime**(1/5)) + 2)]  # calculate up to the next integer
    # plt.xticks(x_ticks, [f'{tick}' for tick in x_ticks])
    # plt.yticks(y_ticks, [f'{tick}' for tick in y_ticks])

    plt.savefig("num_outcomesVSruntime.png")

    shot_r = 4
    random_params_r = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    possible_outcome_r = 8

    input_sizes_r = []
    runtimes_r = [2.107392, 1.724768, 1.716800, 2.619808,
                  6.720928, 39.033825, 292.119080, 2328.342529, 18656.494141]

    for i in range(len(random_params_r)):
        input_sizes_r.append((possible_outcome_r ** random_params_r[i])*shot_r)

    plt.figure()
    plt.plot(input_sizes_r, runtimes_r, marker='o')
    for i, (input_size, runtime) in enumerate(zip(input_sizes_r, runtimes_r)):
        plt.text(input_size, runtime,
                 f'({input_size}, {runtime:.2f})', ha='right', va='bottom')

    plt.xlabel("Input size")
    plt.ylabel("Runtime (ms)")
    plt.title("Runtime vs Input size (Log Scale)")
    plt.grid(True)
    plt.xscale('log')  # Set x-axis to logarithmic scale
    plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.savefig("num_random_varsVSruntime.png")

    shot_s = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    random_params_s = 5
    possible_outcome_s = 8

    input_sizes_s = []
    runtimes_s = [3.358784, 4.315488, 5.128672,
                  7.055616, 8.121024, 10.359488, 10.342336, 11.252736, 13.157984]

    for i in range(len(shot_s)):
        input_sizes_s.append((possible_outcome_s ** random_params_s)*shot_s[i])

    plt.figure()
    plt.plot(input_sizes_s, runtimes_s, marker='o')
    for i, (input_size, runtime) in enumerate(zip(input_sizes_s, runtimes_s)):
        plt.text(input_size, runtime,
                 f'({input_size}, {runtime:.2f})', ha='right', va='bottom')

    plt.xlabel("Input size")
    plt.ylabel("Runtime (ms)")
    plt.title("Runtime vs Input size")
    plt.grid(True)
    plt.savefig("num_shotsVSruntime.png")


if __name__ == "__main__":
    main()
