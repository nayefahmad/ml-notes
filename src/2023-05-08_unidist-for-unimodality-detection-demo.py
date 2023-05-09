# # UniDip algorithim for detecting multimodality

# ## Overview


# ## Conclusions


import warnings
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import weibull_min
from unidip import UniDip

warnings.simplefilter("ignore")

num_datasets_per_case = 10


@dataclass
class SimulationCase:
    shape_01: float
    shape_02: Optional[float]
    num_points_01: int
    num_points_02: Optional[int]
    scale_01: float = 1.0
    scale_02: float = 1.0


examples_case_01 = []
for shape in np.arange(0.2, 2.5, 0.2):
    case = SimulationCase(
        shape_01=shape, shape_02=None, num_points_01=50, num_points_02=None
    )
    examples_case_01.append(case)

examples_case_02 = []
for shape in np.arange(0.2, 2.5, 0.2):
    case = SimulationCase(
        shape_01=shape, shape_02=1.5, num_points_01=50, num_points_02=50
    )
    examples_case_02.append(case)

examples_case_03 = []
for shape in np.arange(0.2, 2.5, 0.2):
    case = SimulationCase(
        shape_01=shape, shape_02=1.5, num_points_01=100, num_points_02=100
    )
    examples_case_03.append(case)

examples_case_04 = []
for shape in np.arange(0.2, 2.5, 0.2):
    case = SimulationCase(
        shape_01=shape,
        shape_02=1.5,
        num_points_01=100,
        num_points_02=100,
        scale_01=1.0,
        scale_02=2.0,
    )
    examples_case_04.append(case)

examples_case_05 = []
for shape in np.arange(0.2, 2.5, 0.2):
    case = SimulationCase(
        shape_01=shape,
        shape_02=1.5,
        num_points_01=100,
        num_points_02=100,
        scale_01=1.0,
        scale_02=10.0,
    )
    examples_case_05.append(case)

examples_case_06 = []
for shape in np.arange(0.2, 2.5, 0.2):
    case = SimulationCase(
        shape_01=shape,
        shape_02=10,
        num_points_01=100,
        num_points_02=100,
        scale_01=1.0,
        scale_02=2.0,
    )
    examples_case_06.append(case)

examples_case_07 = []
for shape in np.arange(0.2, 2.5, 0.2):
    case = SimulationCase(
        shape_01=shape,
        shape_02=10,
        num_points_01=100,
        num_points_02=100,
        scale_01=1.0,
        scale_02=1.0,
    )
    examples_case_07.append(case)

examples_case_08 = []
for shape in np.arange(0.2, 2.5, 0.2):
    case = SimulationCase(
        shape_01=shape,
        shape_02=5,
        num_points_01=100,
        num_points_02=100,
        scale_01=1.0,
        scale_02=1.0,
    )
    examples_case_08.append(case)

examples_case_09 = []
for shape in np.arange(0.2, 2.5, 0.2):
    case = SimulationCase(
        shape_01=shape,
        shape_02=5,
        num_points_01=50,
        num_points_02=50,
        scale_01=1.0,
        scale_02=1.0,
    )
    examples_case_09.append(case)

examples_case_10 = []
for scale in np.arange(1, 10, 1):
    case = SimulationCase(
        shape_01=2.0,
        shape_02=2.0,
        num_points_01=50,
        num_points_02=50,
        scale_01=scale,
        scale_02=1.0,
    )
    examples_case_10.append(case)

all_examples = (
    examples_case_01
    + examples_case_02
    + examples_case_03
    + examples_case_04
    + examples_case_05
    + examples_case_06
    + examples_case_07
    + examples_case_08
    + examples_case_09
    + examples_case_10
)


datasets = {}
results = {}
with PdfPages("2023-05-08_unidip.pdf") as pdf:

    for idx, example in enumerate(all_examples):
        data = weibull_min.rvs(
            c=example.shape_01, scale=example.scale_01, size=example.num_points_01
        )

        # remove extreme outliers
        if example.shape_01 < 1.0:
            q90 = np.quantile(data, 0.9)
            data = data[data < q90]

        txt = f"idx={idx} \nshape 1={example.shape_01:.2f}, scale 1={example.scale_01},"
        txt += f" n01={example.num_points_01}"

        if example.shape_02 is not None:
            data2 = weibull_min.rvs(
                c=example.shape_02, scale=example.scale_02, size=example.num_points_02
            )
            data = np.concatenate([data, data2])
            txt += f"\nshape 2={example.shape_02:.2f}, scale 1={example.scale_02}, "
            txt += f"n02={example.num_points_02}"

        # unidip algorithm:
        data = np.msort(data)
        intervals = UniDip(data).run()
        print(f"idx={idx}; intervals: {intervals}")
        num_modes = len(intervals)

        plot_vlines = []
        while len(intervals) > 0:
            first_interval = intervals[0]
            lower = first_interval[0]
            upper = first_interval[1]
            plot_vlines.append(lower)
            plot_vlines.append(upper)
            intervals = intervals[1:]

        print_plots = True
        if print_plots:
            txt += f"\nnum peaks found={num_modes}"
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.distplot(data, ax=ax)
            plt.title(txt)
            for vline in plot_vlines:
                plt.axvline(x=data[vline], color="red")
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        datasets[idx] = data

        results[idx] = {
            "c01": example.shape_01,
            "scale01": example.scale_01,
            "n01": example.num_points_01,
            "c02": example.shape_02,
            "scale02": example.scale_02,
            "n02": example.num_points_02,
            "estimated_num_modes": num_modes,
        }

results_df = pd.DataFrame(results).T
results_df.index.rename("idx", inplace=True)
print(results_df)


# for ad-hoc exploration:
idx_selected = 60
np.quantile(datasets[idx_selected], [0.9, 0.95, 0.99, 1.0])
data = datasets[idx_selected]
q90 = np.quantile(data, 0.90)
data = data[data < q90]
sns.distplot(data)
plt.show()
